import torch
from skimage.transform import resize
from diffusion import create_diffusion
from diffusers.models import AutoencoderKL
from models import UNET_models
import argparse
import numpy as np
torch.set_grad_enabled(False)
import torch.nn.functional as F
device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    print("GPU not found. Using CPU instead.")
from glob import glob
import os
from sklearn.metrics import precision_recall_curve
import matplotlib.pyplot as plt
import random

from torch.utils.data import DataLoader
from torchvision import transforms
from KvaDataLoader import KvasirDataset
from MVTECDataLoader import MVTECDataset
from VISADataLoader import VISADataset
from BRATDataLoader import BRATDataset
from scipy.ndimage import gaussian_filter

from sklearn.metrics import average_precision_score
from skimage.metrics import structural_similarity as ssim
from numpy import ndarray
import pandas as pd
from skimage import measure
from sklearn.metrics import auc

def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """


    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        df = pd.concat([df, pd.DataFrame({"pro": [np.mean(pros)], "fpr": [fpr], "threshold": [th]})], ignore_index=True)

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def find_best_threshold(gt, score):
    gt_flat = gt.astype(np.uint8).ravel()
    sc_flat = score.astype(np.float32).ravel()
    precision, recall, thresholds = precision_recall_curve(gt_flat, sc_flat)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1))
    return float(thresholds[best_idx]), float(f1[best_idx])

def binarize_with_threshold(maps: np.ndarray, threshold: float):
    """
    maps: (N,H,W) 连续的 anomaly map
    threshold: 标量
    return: (N,H,W) 0/1 二值图
    """
    return (maps > threshold).astype(np.uint8)

def dice_at_threshold(ground_truth: np.ndarray, maps: np.ndarray, threshold: float):
    bin_pred = binarize_with_threshold(maps, threshold)
    return dice_score(bin_pred, ground_truth)  # 直接复用你已有的 dice_score


def dice_score(pred, gt, smooth=1e-6):
    """
    计算 Dice 系数
    pred, gt: numpy arrays of shape (N, H, W)
    """
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0.5).astype(np.uint8)
    intersection = np.sum(pred * gt)
    return (2. * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)

def sensitivity_score(pred, gt, smooth=1e-6):
    """
    计算 Sensitivity / Recall
    pred, gt: numpy arrays of shape (N, H, W)
    """
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0.5).astype(np.uint8)

    tp = np.sum((pred == 1) & (gt == 1))
    fn = np.sum((pred == 0) & (gt == 1))

    return (tp + smooth) / (tp + fn + smooth)

def ssim_score(pred, gt):
    """
    计算平均 SSIM
    pred, gt: numpy arrays of shape (N, H, W)
    这里按二值 mask 和 GT mask 来算
    """
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0.5).astype(np.uint8)

    scores = []
    for i in range(len(pred)):
        scores.append(ssim(gt[i], pred[i], data_range=1))
    return float(np.mean(scores))


def calculate_metrics(ground_truth, prediction, threshold=0.5):
    """
    只计算:
    1. Dice(F1)      -> 基于 threshold 二值化后的 mask
    2. AP            -> 基于连续 anomaly map
    3. Sensitivity   -> 基于 threshold 二值化后的 mask
    4. SSIM          -> 基于 threshold 二值化后的 mask 与 GT
    """
    flat_gt = ground_truth.flatten().astype(np.uint8)
    flat_pred = prediction.flatten().astype(np.float32)

    # AP: 用连续 anomaly map
    ap = average_precision_score(flat_gt, flat_pred)

    # 二值预测
    binary_pred = (prediction > threshold).astype(np.uint8)

    # Dice(F1)
    dice = dice_score(binary_pred, ground_truth)

    # Sensitivity
    sensitivity = sensitivity_score(binary_pred, ground_truth)

    # SSIM
    ssim_val = ssim_score(binary_pred, ground_truth)

    return dice, ap, sensitivity, ssim_val

def latent_to_grayscale(latents: torch.Tensor, out_h: int, out_w: int) -> np.ndarray:
    z = latents.detach().cpu()
    gray = torch.norm(z, dim=0).numpy()

    # ---------- 分位数拉伸 ----------
    p1, p99 = np.percentile(gray, (1, 99))
    gray = np.clip((gray - p1) / (p99 - p1 + 1e-8), 0, 1)

    # ---------- Gamma增强 ----------
    gamma = 0.5
    gray = gray ** gamma

    # ---------- resize ----------
    gray = F.interpolate(
        torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).float(),
        size=(out_h, out_w),
        mode="nearest"
    )[0, 0].numpy()

    return (gray * 255).astype(np.uint8)

def latent_pca_blocky(latents: torch.Tensor, out_h: int, out_w: int) -> np.ndarray:
    """
    对 latent 做 PCA->3通道，再最近邻上采样。
    """
    z = latents.detach().cpu()   # (C,h,w)
    C, h, w = z.shape
    assert C >= 3, "latent 通道数至少要 >=3"
    z = z.reshape(C, -1).T  # (HW,C)
    z = z - z.mean(dim=0, keepdim=True)
    cov = z.T @ z / (z.shape[0] - 1)
    _, eigvecs = torch.linalg.eigh(cov)  # (C,C)
    comps = eigvecs[:, -3:]  # 取前三主成分
    y = z @ comps  # (HW,3)
    y_min, y_max = y.min(0, keepdim=True)[0], y.max(0, keepdim=True)[0]
    y = (y - y_min) / (y_max - y_min + 1e-8)
    y = y.T.reshape(3, h, w).unsqueeze(0)  # (1,3,h,w)
    y = F.interpolate(y, size=(out_h, out_w), mode="nearest")
    y = (y[0].permute(1,2,0).numpy() * 255).astype(np.uint8)  # (H,W,3)
    return y

def smooth_mask(mask, sigma=1.0):
    smoothed_mask = gaussian_filter(mask, sigma=sigma)
    return smoothed_mask

def visualize_with_vae_results(input_images, vae_inputs, vae_latent_pred,
                               output_images, vae_latent_recons,
                               anomaly_map, gts, best_thresh,
                               save_dir, num_samples=5):
    """
    可视化六张图：
    1. 原图
    2. 原图 latent
    3. 预测 latent
    4. Reconstruction (decode 后)
    5. Reconstruction latent
    6. Binary Mask
    """
    os.makedirs(save_dir, exist_ok=True)
    indices = random.sample(range(len(input_images)), min(num_samples, len(input_images)))

    for count, i in enumerate(indices):
        fig, axs = plt.subplots(1, 6, figsize=(22, 4))

        axs[0].imshow(input_images[i])
        axs[0].set_title("Original"); axs[0].axis("off")

        axs[1].imshow(vae_inputs[i], cmap='gray')  # 原图 latent
        axs[1].set_title("Original Latent"); axs[1].axis("off")

        axs[2].imshow(vae_latent_pred[i], cmap='gray')  # 预测 latent
        axs[2].set_title("Predicted Latent"); axs[2].axis("off")

        axs[3].imshow(output_images[i])     # Reconstruction (RGB)
        axs[3].set_title("Reconstruction"); axs[3].axis("off")

        axs[4].imshow(vae_latent_recons[i], cmap='gray')
        axs[4].set_title("Reconstruction Latent"); axs[4].axis("off")

        bin_mask = (anomaly_map[i] > best_thresh).astype(np.uint8)
        axs[5].imshow(bin_mask, cmap='gray')
        axs[5].set_title(f"Binary Mask\n@ {best_thresh:.3f}"); axs[5].axis("off")

        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"vae_result_{count}.png"), bbox_inches='tight')
        plt.close()


def latent_to_vis(z, size=256):
    """
    把 latent 转成可视化图:
    - 计算 L2 norm 热图
    """
    z = z.detach().cpu()
    norm = torch.norm(z, dim=0).numpy()  # (h,w)
    norm = (norm - norm.min()) / (norm.max() - norm.min() + 1e-8)
    norm = resize(norm, (size, size))
    return (norm * 255).astype(np.uint8)

    

def calculate_anomaly_maps(x0_s, encoded_s,  image_samples_s, latent_samples_s, center_size=256):
    pred_geometric = []
    pred_aritmetic = []
    image_differences = []
    latent_differences = []
    input_images = []
    output_images = []
    for x, encoded,  image_samples, latent_samples in zip(x0_s, encoded_s,  image_samples_s, latent_samples_s):
            
        input_image = ((np.clip(x[0].detach().cpu().numpy(), -1, 1).transpose(1,2,0))*127.5+127.5).astype(np.uint8)
        output_image = ((np.clip(image_samples[0].detach().cpu().numpy(), -1, 1).transpose(1,2,0))*127.5+127.5).astype(np.uint8)
        input_images.append(input_image)
        output_images.append(output_image)

        image_difference = (((((torch.abs(image_samples-x))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).max(axis=2))
        image_difference = (np.clip(image_difference, 0.0, 0.4) ) * 2.5
        image_difference = smooth_mask(image_difference, sigma=3)
        image_differences.append(image_difference)
        
        latent_difference = (((((torch.abs(latent_samples-encoded))).to(torch.float32)).mean(axis=0)).detach().cpu().numpy().transpose(1,2,0).mean(axis=2))
        latent_difference = (np.clip(latent_difference, 0.0 , 0.2)) * 5
        latent_difference = smooth_mask(latent_difference, sigma=1)
        latent_difference = resize(latent_difference, (center_size, center_size))
        latent_differences.append(latent_difference)
        
        final_anomaly = image_difference * latent_difference
        final_anomaly = np.sqrt(final_anomaly)
        final_anomaly2 = 1/2*image_difference + 1/2*latent_difference
        pred_geometric.append(final_anomaly)
        pred_aritmetic.append(final_anomaly2)
            
    pred_geometric = np.stack(pred_geometric, axis=0)
    pred_aritmetic = np.stack(pred_aritmetic, axis=0)
    latent_differences = np.stack(latent_differences, axis=0)
    image_differences = np.stack(image_differences, axis=0)

    return {'anomaly_geometric':pred_geometric, 'anomaly_aritmetic':pred_aritmetic, 'latent_discrepancy':latent_differences, 'image_discrepancy':image_differences}, input_images, output_images



def evaluate_anomaly_maps(anomaly_maps, segmentation):
    for key in anomaly_maps.keys():
        auroc_score, aupro_score, f1_max_score, ap, aurocsp, apsp, f1sp, dice = calculate_metrics(segmentation, anomaly_maps[key])
        auroc_score, aupro_score, f1_max_score, ap, aurocsp, apsp, f1sp, dice = \
            np.round(auroc_score, 4), np.round(aupro_score, 4), np.round(f1_max_score, 4), \
            np.round(ap, 4), np.round(aurocsp, 4), np.round(apsp, 4), np.round(f1sp, 4), np.round(dice, 4)
        
        print(f'{key}: auroc:{auroc_score:.4f}, aupro:{aupro_score:.4f}, f1_max:{f1_max_score:.4f}, '
              f'ap:{ap:.4f}, aurocsp:{aurocsp:.4f}, apsp:{apsp:.4f}, f1sp:{f1sp:.4f}, dice:{dice:.4f}')


def evaluation(args):
    # vae_model = f"./sdxl-vae-fp16-fix" #@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
    vae_model = f"stabilityai/sd-vae-ft-{args.vae_type}" #@param ["stabilityai/sd-vae-ft-mse", "stabilityai/sd-vae-ft-ema"]
    vae = AutoencoderKL.from_pretrained(vae_model).to(device)
    vae.eval()
    try:
        if args.model_path != '':
            ckpt = args.model_path
        else:
            path = f"./DeCo-Diff_{args.dataset}_{args.object_category}_{args.model_size}_{args.center_size}"
            try:
                ckpt = sorted(glob(f'{path}/last.pt'))[-1]
            except:
                ckpt = sorted(glob(f'{path}/*/last.pt'))[-1]
    except:
        raise Exception("Please provide the trained model's path using --model_path")
    

    latent_size = int(args.center_size) // 8
    model = UNET_models[args.model_size](latent_size=latent_size, ncls=args.num_classes)
    
    state_dict = torch.load(ckpt)['model']
    print(model.load_state_dict(state_dict))
    model.eval() # important!
    model.to(device)
    print('model loaded')


    print('=='*30)
    print('Starting Evaluation...')
    print('=='*30)

    metric_collector = {
        "dice": [],
        "ap": [],
        "sensitivity": [],
        "ssim": []
    }

    for category in args.categories:


        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
        ])
            
        # Create diffusion object:
        diffusion = create_diffusion(f'ddim{args.reverse_steps}', predict_deviation=False, sigma_small=False, predict_xstart=False, diffusion_steps=10)
            

        encoded_s = []
        image_samples_s = []
        latent_samples_s = []
        x0_s = []
        segmentation_s = []
        
        if args.dataset=='mvtec':
            test_dataset = MVTECDataset('test', object_class=category, rootdir=args.data_dir, transform=transform, normal=False, anomaly_class=args.anomaly_class, image_size=args.image_size, center_size=args.actual_image_size, center_crop=args.center_crop)
        elif args.dataset=='visa':
            test_dataset = VISADataset('test', object_class=category, rootdir=args.data_dir, transform=transform, normal=False, anomaly_class=args.anomaly_class, image_size=args.image_size, center_size=args.actual_image_size, center_crop=args.center_crop)
        elif args.dataset == 'brat':
            test_dataset = BRATDataset('test', object_class=category, rootdir=args.data_dir, transform=transform, image_size=args.image_size, center_size=args.actual_image_size, center_crop=args.center_crop)        
        elif args.dataset == 'kvasir':
            test_dataset = KvasirDataset('test', object_class=category, rootdir=args.data_dir, transform=transform, image_size=args.image_size, center_size=args.actual_image_size, center_crop=args.center_crop)

        test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False, num_workers=4, drop_last=False)

        vae_input_images = []
        vae_recons_input_images = []
        vae_recons_output_images = []
        vae_latent_pred_images = []
        vae_latent_recons_images = []


        for ii, (x, seg, object_cls) in enumerate(test_loader):
            with torch.no_grad():
                # ========== VAE encode 原图 ========== 
                encoded = vae.encode(x.to(device)).latent_dist.mean.mul_(0.18215)

                model_kwargs = {
                    'context': object_cls.to(device).unsqueeze(1),
                    'mask': None
                }
                latent_samples = diffusion.ddim_deviation_sample_loop(
                    model, encoded.shape, noise=encoded, clip_denoised=False,
                    start_t=args.reverse_steps,
                    model_kwargs=model_kwargs, progress=False, device=device,
                    eta=0
                )

                # 解码
                image_samples = vae.decode(latent_samples / 0.18215).sample   # 模型重构
                x0 = vae.decode(encoded / 0.18215).sample                     # 原图 VAE 重建

                # ===== VAE 可视化保存 =====
                batch_size = x.size(0)

                for b in range(batch_size):
                    # (1) 原图 latent
                    # z_vis = latent_pca_blocky(encoded[b], args.center_size, args.center_size)
                    z_vis = latent_to_grayscale(encoded[b], args.center_size, args.center_size)
                    vae_input_images.append(z_vis)

                    # (2) Predicted DoD = latent_samples - encoded
                    diff_latent = latent_samples[b] - encoded[b]
                    # 差分可视化（PCA 彩色块）
                    # z_vis = latent_pca_blocky(diff_latent, args.center_size, args.center_size)
                    z_vis = latent_to_grayscale(diff_latent, args.center_size, args.center_size)
                    vae_latent_pred_images.append(z_vis)

                    # (3) Reconstruction (decode 后 RGB)
                    _x0 = x0[b]
                    vae_recons_input_images.append(
                        ((np.clip(_x0.detach().cpu().numpy().transpose(1, 2, 0), -1, 1) * 127.5 + 127.5).astype(np.uint8))
                    )

                    # (4) Reconstruction latent
                    z_out = vae.encode(image_samples[b:b+1]).latent_dist.mean.mul_(0.18215)
                    # z_vis = latent_pca_blocky(z_out[0], args.center_size, args.center_size)
                    z_vis = latent_to_grayscale(z_out[0], args.center_size, args.center_size)
                    vae_latent_recons_images.append(z_vis)


            segmentation_s += [_seg.squeeze() for _seg in seg]
            encoded_s += [_encoded.unsqueeze(0) for _encoded in encoded]
            image_samples_s += [_image_samples.unsqueeze(0) for _image_samples in image_samples]
            latent_samples_s += [_latent_samples.unsqueeze(0) for _latent_samples in latent_samples]
            x0_s += [_x0.unsqueeze(0) for _x0 in x0]

        print(category)        
        anomaly_maps, input_images, output_images = calculate_anomaly_maps(
            x0_s, encoded_s, image_samples_s, latent_samples_s, center_size=args.center_size
        )

        seg_np = np.stack(segmentation_s, axis=0).astype(np.uint8)


        # 用 anomaly_geometric 找最佳阈值
        key_main = 'anomaly_geometric'
        best_thresh, best_f1 = find_best_threshold(seg_np, anomaly_maps[key_main])
        print(f"[{category}] Best Threshold: {best_thresh:.6f}, F1@Best: {best_f1:.4f}")

        # 在最佳阈值下二值化 anomaly map
        binary_pred = binarize_with_threshold(anomaly_maps[key_main], best_thresh)

        print(">>> DEBUG: GT mask unique values:", np.unique(seg_np))
        print(">>> DEBUG: anomaly map min/max:", anomaly_maps[key_main].min(), anomaly_maps[key_main].max())


        dice, ap, sensitivity, ssim_val = \
            calculate_metrics(seg_np, anomaly_maps[key_main], threshold=best_thresh)

        # ===== 收集指标，用于 avg =====
        metric_collector["dice"].append(dice)
        metric_collector["ap"].append(ap)
        metric_collector["sensitivity"].append(sensitivity)
        metric_collector["ssim"].append(ssim_val)

        print(f"[{category}] Metrics @ Best Threshold:")
        print(f"    Dice(F1):    {dice:.4f}")
        print(f"    AP:          {ap:.4f}")
        print(f"    Sensitivity: {sensitivity:.4f}")
        print(f"    SSIM:        {ssim_val:.4f}")


        save_path = "/home/shengfeng.ye22/DeCo-Diff/img"
        visualize_with_vae_results(
            input_images,
            vae_input_images,          # 原图 latent
            vae_latent_pred_images,    # 预测 latent
            output_images,             # Reconstruction (RGB)
            vae_latent_recons_images,  # 重建 latent
            anomaly_maps[key_main],
            seg_np,
            best_thresh,
            save_dir=save_path,     
            num_samples=20
        )



        print('=='*30)

    print("\n" + "="*30)
    print(">>> Average Metrics Across Categories")
    print("="*30)

    def safe_mean(x):
        x = np.array(x, dtype=np.float32)
        return np.nanmean(x)

    print(f"Avg Dice(F1):    {safe_mean(metric_collector['dice']):.4f}")
    print(f"Avg AP:          {safe_mean(metric_collector['ap']):.4f}")
    print(f"Avg Sensitivity: {safe_mean(metric_collector['sensitivity']):.4f}")
    print(f"Avg SSIM:        {safe_mean(metric_collector['ssim']):.4f}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, choices=['brat'], default="brat")
    parser.add_argument("--data-dir", type=str, default='./data/')
    parser.add_argument("--model-size", type=str, choices=['UNet_XS','UNet_S', 'UNet_M', 'UNet_L', 'UNet_XL'], default='UNet_M')
    parser.add_argument("--image-size", type=int, default= 288)
    parser.add_argument("--center-size", type=int, default=256)
    parser.add_argument("--center-crop", type=lambda v: True if v.lower() in ('yes','true','t','y','1') else False, default=True)
    parser.add_argument("--vae-type", type=str, choices=["ema", "mse"], default="ema") 
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--object-category", type=str, default='all')
    parser.add_argument("--model-path", type=str, default='.')
    parser.add_argument("--anomaly-class", type=str, default='all')
    parser.add_argument("--reverse-steps", type=int, default=5)
    
    
    args = parser.parse_args()
    if args.dataset == 'brat':
        args.num_classes = 1
    args.results_dir = f"./STA-Diff_{args.dataset}_{args.object_category}_{args.model_size}_{args.center_size}"
    if args.center_crop:
        args.results_dir += "_CenterCrop"
        args.actual_image_size = args.center_size
    else:
        args.actual_image_size = args.image_size
    args.categories = [args.object_category]
        
    evaluation(args)
