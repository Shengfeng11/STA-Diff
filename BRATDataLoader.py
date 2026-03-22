import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from glob import glob

class BRATDataset(Dataset):
    """
    三模态 BraTS 数据集：
        T1 / T2 / FLAIR 输入作为 3 通道
        seg 作为二值 mask

    文件结构示例:
        root/
            test/
                T1/
                    BraTS2021_00072_T1_76.npy
                T2/
                    BraTS2021_00072_T2_76.npy
                FLAIR/
                    BraTS2021_00072_FLAIR_76.npy
                seg/
                    BraTS2021_00072_seg_76.npy
    """

    def __init__(
        self,
        mode: str,
        object_class: str,
        transform=None,
        # modalities=("flair", "t2", "t1ce"),   # ← 多模态
        modalities=("t1ce", "t1ce", "t1ce"),   # ← 多模态
        rootdir="./data",
        image_size=240,
        center_size=240,
        augment=False,
        center_crop=False,
        debug_first_n=0,
    ):
        self.mode = mode
        self.modalities = modalities
        self.image_size = image_size
        self.center_size = center_size
        self.center_crop = center_crop
        self.debug_first_n = debug_first_n
        self.transform = transform

        # 模态路径
        self.modality_dirs = {
            m: os.path.join(rootdir, mode, m) for m in modalities
        }
        self.mask_dir = os.path.join(rootdir, mode, "seg")

        # —— 以第一个模态(T1)为主索引 —— #
        main_mod = modalities[0]
        self.main_paths = sorted(
            glob(os.path.join(self.modality_dirs[main_mod], "*.npy"))
        )
        if len(self.main_paths) == 0:
            raise RuntimeError(f"No modality files found in {self.modality_dirs[main_mod]}")

        # 匹配所有模态 & mask
        self.sample_list = []
        for p in self.main_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            # 例：BraTS2021_00072_T1_76
            parts = stem.split("_")
            prefix = "_".join(parts[:2])         # BraTS2021_00072
            slice_id = parts[-1]                # 76

            # 构造每个模态的文件路径
            modal_files = {}
            ok = True
            for m in modalities:
                modal_name = f"{prefix}_{m}_{slice_id}.npy"
                modal_path = os.path.join(self.modality_dirs[m], modal_name)
                if not os.path.exists(modal_path):
                    ok = False
                    break
                modal_files[m] = modal_path

            if not ok:
                continue

            # 构造 mask path
            mask_name = f"{prefix}_seg_{slice_id}.npy"
            mask_path = os.path.join(self.mask_dir, mask_name)
            if not os.path.exists(mask_path):
                continue

            self.sample_list.append((modal_files, mask_path))

        print(f"[BRATDataset Multi-modal] mode={mode} samples={len(self.sample_list)}")

        # 数据增强
        if augment:
            self.aug = A.Compose([
                A.Affine(translate_px=int(image_size/8 - center_size/8), p=0.5),
                A.CenterCrop(p=1, height=center_size, width=center_size),
            ])
        else:
            self.aug = A.CenterCrop(p=1, height=center_size, width=center_size)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        modal_files, mask_path = self.sample_list[idx]

        # 读取三模态并 resize
        channels = []
        for m in self.modalities:
            x = np.load(modal_files[m]).astype(np.float32)
            if x.ndim == 3:
                x = x.squeeze()
            x = cv2.resize(x, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR)
            channels.append(x)

        img = np.stack(channels, axis=2)   # (H,W,3)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # normalize 0~1

        # mask
        seg = np.load(mask_path).astype(np.uint8)
        if seg.ndim == 3:
            seg = seg.squeeze()
        seg = cv2.resize(seg, (self.image_size, self.image_size), interpolation=cv2.INTER_NEAREST)
        seg = (seg > 0).astype(np.uint8)

        # augmentation
        if self.center_crop:
            out = self.aug(image=img, mask=seg)
            img, seg = out["image"], out["mask"]

        img = torch.from_numpy(img.transpose(2, 0, 1))   # (3,H,W)
        seg = torch.from_numpy(seg)

        return img, seg, 0
