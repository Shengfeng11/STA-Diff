import torch
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
from torch.utils.data import DataLoader
from torchvision import transforms
from BRATDataLoader import BRATDataset
from scipy.ndimage import gaussian_filter
from sklearn.metrics import average_precision_score
from skimage.metrics import structural_similarity as ssim
from numpy import ndarray
import pandas as pd
from skimage import measure
from sklearn.metrics import auc


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> float:
    """
    Compute the area under the PRO curve within the FPR range [0, 0.3].

    Args:
        masks: Binary ground-truth masks of shape (N, H, W).
        amaps: Continuous anomaly maps of shape (N, H, W).
        num_th: Number of thresholds.

    Returns:
        PRO-AUC score.
    """
    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (N, H, W)"
    assert masks.ndim == 3, "masks.ndim must be 3 (N, H, W)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must match"
    assert set(masks.flatten()) == {0, 1}, "masks must be binary"
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

        df = pd.concat(
            [df, pd.DataFrame({"pro": [np.mean(pros)], "fpr": [fpr], "threshold": [th]})],
            ignore_index=True
        )

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
    Convert continuous anomaly maps into binary masks.

    Args:
        maps: Continuous anomaly maps of shape (N, H, W).
        threshold: Scalar threshold.

    Returns:
        Binary masks of shape (N, H, W).
    """
    return (maps > threshold).astype(np.uint8)


def dice_at_threshold(ground_truth: np.ndarray, maps: np.ndarray, threshold: float):
    bin_pred = binarize_with_threshold(maps, threshold)
    return dice_score(bin_pred, ground_truth)


def dice_score(pred, gt, smooth=1e-6):
    """
    Compute Dice score.

    Args:
        pred: Predicted masks of shape (N, H, W).
        gt: Ground-truth masks of shape (N, H, W).

    Returns:
        Dice coefficient.
    """
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0.5).astype(np.uint8)
    intersection = np.sum(pred * gt)
    return (2. * intersection + smooth) / (np.sum(pred) + np.sum(gt) + smooth)


def sensitivity_score(pred, gt, smooth=1e-6):
    """
    Compute sensitivity (recall).

    Args:
        pred: Predicted masks of shape (N, H, W).
        gt: Ground-truth masks of shape (N, H, W).

    Returns:
        Sensitivity score.
    """
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0.5).astype(np.uint8)

    tp = np.sum((pred == 1) & (gt == 1))
    fn = np.sum((pred == 0) & (gt == 1))

    return (tp + smooth) / (tp + fn + smooth)


def ssim_score(pred, gt):
    """
    Compute mean SSIM between predicted binary masks and ground-truth masks.

    Args:
        pred: Predicted masks of shape (N, H, W).
        gt: Ground-truth masks of shape (N, H, W).

    Returns:
        Mean SSIM score.
    """
    pred = (pred > 0.5).astype(np.uint8)
    gt = (gt > 0.5).astype(np.uint8)

    scores = []
    for i in range(len(pred)):
        scores.append(ssim(gt[i], pred[i], data_range=1))
    return float(np.mean(scores))


def calculate_metrics(ground_truth, prediction, threshold=0.5):
    """
    Compute evaluation metrics:
    1. Dice (F1) based on thresholded binary masks
    2. AP based on continuous anomaly maps
    3. Sensitivity based on thresholded binary masks
    4. SSIM based on thresholded binary masks
    """
    flat_gt = ground_truth.flatten().astype(np.uint8)
    flat_pred = prediction.flatten().astype(np.float32)

    ap = average_precision_score(flat_gt, flat_pred)

    binary_pred = (prediction > threshold).astype(np.uint8)

    dice = dice_score(binary_pred, ground_truth)
    sensitivity = sensitivity_score(binary_pred, ground_truth)
    ssim_val = ssim_score(binary_pred, ground_truth)

    return dice, ap, sensitivity, ssim_val


def smooth_mask(mask, sigma=1.0):
    return gaussian_filter(mask, sigma=sigma)


def calculate_anomaly_maps(x0_s, encoded_s, image_samples_s, latent_samples_s, center_size=256):
    pred_geometric = []
    pred_arithmetic = []
    image_differences = []
    latent_differences = []
    input_images = []
    output_images = []

    for x, encoded, image_samples, latent_samples in zip(x0_s, encoded_s, image_samples_s, latent_samples_s):
        input_image = ((np.clip(x[0].detach().cpu().numpy(), -1, 1).transpose(1, 2, 0)) * 127.5 + 127.5).astype(np.uint8)
        output_image = ((np.clip(image_samples[0].detach().cpu().numpy(), -1, 1).transpose(1, 2, 0)) * 127.5 + 127.5).astype(np.uint8)
        input_images.append(input_image)
        output_images.append(output_image)

        image_difference = (
            ((((torch.abs(image_samples - x))).to(torch.float32)).mean(axis=0))
            .detach().cpu().numpy().transpose(1, 2, 0).max(axis=2)
        )
        image_difference = (np.clip(image_difference, 0.0, 0.4)) * 2.5
        image_difference = smooth_mask(image_difference, sigma=3)
        image_differences.append(image_difference)

        latent_difference = (
            ((((torch.abs(latent_samples - encoded))).to(torch.float32)).mean(axis=0))
            .detach().cpu().numpy().transpose(1, 2, 0).mean(axis=2)
        )
        latent_difference = (np.clip(latent_difference, 0.0, 0.2)) * 5
        latent_difference = smooth_mask(latent_difference, sigma=1)
        latent_difference = F.interpolate(
            torch.tensor(latent_difference).unsqueeze(0).unsqueeze(0).float(),
            size=(center_size, center_size),
            mode="nearest"
        )[0, 0].numpy()
        latent_differences.append(latent_difference)

        final_anomaly = image_difference * latent_difference
        final_anomaly = np.sqrt(final_anomaly)
        final_anomaly2 = 0.5 * image_difference + 0.5 * latent_difference

        pred_geometric.append(final_anomaly)
        pred_arithmetic.append(final_anomaly2)

    pred_geometric = np.stack(pred_geometric, axis=0)
    pred_arithmetic = np.stack(pred_arithmetic, axis=0)
    latent_differences = np.stack(latent_differences, axis=0)
    image_differences = np.stack(image_differences, axis=0)

    return {
        'anomaly_geometric': pred_geometric,
        'anomaly_arithmetic': pred_arithmetic,
        'latent_discrepancy': latent_differences,
        'image_discrepancy': image_differences
    }, input_images, output_images


def evaluation(args):
    vae_model = f"stabilityai/sd-vae-ft-{args.vae_type}"
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
        raise Exception("Please provide the trained model path using --model_path")

    latent_size = int(args.center_size) // 8
    model = UNET_models[args.model_size](latent_size=latent_size, ncls=args.num_classes)

    state_dict = torch.load(ckpt)['model']
    print(model.load_state_dict(state_dict))
    model.eval()
    model.to(device)
    print('Model loaded')

    print('==' * 30)
    print('Starting evaluation...')
    print('==' * 30)

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

        diffusion = create_diffusion(
            f'ddim{args.reverse_steps}',
            predict_deviation=False,
            sigma_small=False,
            predict_xstart=False,
            diffusion_steps=10
        )

        encoded_s = []
        image_samples_s = []
        latent_samples_s = []
        x0_s = []
        segmentation_s = []

        if args.dataset == 'brat':
            test_dataset = BRATDataset(
                'test',
                object_class=category,
                rootdir=args.data_dir,
                transform=transform,
                image_size=args.image_size,
                center_size=args.actual_image_size,
                center_crop=args.center_crop
            )

        test_loader = DataLoader(
            test_dataset,
            batch_size=8,
            shuffle=False,
            num_workers=4,
            drop_last=False
        )

        for ii, (x, seg, object_cls) in enumerate(test_loader):
            with torch.no_grad():
                encoded = vae.encode(x.to(device)).latent_dist.mean.mul_(0.18215)

                model_kwargs = {
                    'context': object_cls.to(device).unsqueeze(1),
                    'mask': None
                }

                latent_samples = diffusion.ddim_deviation_sample_loop(
                    model,
                    encoded.shape,
                    noise=encoded,
                    clip_denoised=False,
                    start_t=args.reverse_steps,
                    model_kwargs=model_kwargs,
                    progress=False,
                    device=device,
                    eta=0
                )

                image_samples = vae.decode(latent_samples / 0.18215).sample
                x0 = vae.decode(encoded / 0.18215).sample

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

        key_main = 'anomaly_geometric'
        best_thresh, best_f1 = find_best_threshold(seg_np, anomaly_maps[key_main])
        print(f"[{category}] Best threshold: {best_thresh:.6f}, F1@Best: {best_f1:.4f}")

        print(">>> DEBUG: GT mask unique values:", np.unique(seg_np))
        print(">>> DEBUG: anomaly map min/max:", anomaly_maps[key_main].min(), anomaly_maps[key_main].max())

        dice, ap, sensitivity, ssim_val = calculate_metrics(
            seg_np,
            anomaly_maps[key_main],
            threshold=best_thresh
        )

        metric_collector["dice"].append(dice)
        metric_collector["ap"].append(ap)
        metric_collector["sensitivity"].append(sensitivity)
        metric_collector["ssim"].append(ssim_val)

        print(f"[{category}] Metrics @ Best Threshold:")
        print(f"    Dice(F1):    {dice:.4f}")
        print(f"    AP:          {ap:.4f}")
        print(f"    Sensitivity: {sensitivity:.4f}")
        print(f"    SSIM:        {ssim_val:.4f}")

        print('==' * 30)

    print("\n" + "=" * 30)
    print("Average Metrics Across Categories")
    print("=" * 30)

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
    parser.add_argument("--model-size", type=str, choices=['UNet_XS', 'UNet_S', 'UNet_M', 'UNet_L', 'UNet_XL'], default='UNet_M')
    parser.add_argument("--image-size", type=int, default=288)
    parser.add_argument("--center-size", type=int, default=256)
    parser.add_argument("--center-crop", type=lambda v: True if v.lower() in ('yes', 'true', 't', 'y', '1') else False, default=True)
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
