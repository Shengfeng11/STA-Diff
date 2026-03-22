import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from glob import glob


class BRATDataset(Dataset):
    """
    Multi-modal BraTS Dataset:
        Uses T1 / T2 / FLAIR (or customized modalities) as 3-channel input,
        and segmentation (seg) as a binary mask.

    Expected directory structure:
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
        # modalities=("flair", "t2", "t1ce"),   # multi-modal setting
        modalities=("t1ce", "t1ce", "t1ce"),   # duplicated modality (pseudo-RGB)
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

        # Build modality directories
        self.modality_dirs = {
            m: os.path.join(rootdir, mode, m) for m in modalities
        }
        self.mask_dir = os.path.join(rootdir, mode, "seg")

        # Use the first modality as the index reference
        main_mod = modalities[0]
        self.main_paths = sorted(
            glob(os.path.join(self.modality_dirs[main_mod], "*.npy"))
        )

        if len(self.main_paths) == 0:
            raise RuntimeError(
                f"No modality files found in {self.modality_dirs[main_mod]}"
            )

        # Match all modalities and corresponding mask
        self.sample_list = []
        for p in self.main_paths:
            stem = os.path.splitext(os.path.basename(p))[0]
            # Example: BraTS2021_00072_T1_76
            parts = stem.split("_")
            prefix = "_".join(parts[:2])   # BraTS2021_00072
            slice_id = parts[-1]           # 76

            # Construct paths for each modality
            modal_files = {}
            valid = True
            for m in modalities:
                modal_name = f"{prefix}_{m}_{slice_id}.npy"
                modal_path = os.path.join(self.modality_dirs[m], modal_name)
                if not os.path.exists(modal_path):
                    valid = False
                    break
                modal_files[m] = modal_path

            if not valid:
                continue

            # Construct mask path
            mask_name = f"{prefix}_seg_{slice_id}.npy"
            mask_path = os.path.join(self.mask_dir, mask_name)
            if not os.path.exists(mask_path):
                continue

            self.sample_list.append((modal_files, mask_path))

        print(f"[BRATDataset Multi-modal] mode={mode}, samples={len(self.sample_list)}")

        # Data augmentation
        if augment:
            self.aug = A.Compose([
                A.Affine(translate_px=int(image_size / 8 - center_size / 8), p=0.5),
                A.CenterCrop(p=1, height=center_size, width=center_size),
            ])
        else:
            self.aug = A.CenterCrop(p=1, height=center_size, width=center_size)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        modal_files, mask_path = self.sample_list[idx]

        # Load multi-modal inputs and resize
        channels = []
        for m in self.modalities:
            x = np.load(modal_files[m]).astype(np.float32)
            if x.ndim == 3:
                x = x.squeeze()

            x = cv2.resize(
                x,
                (self.image_size, self.image_size),
                interpolation=cv2.INTER_LINEAR
            )
            channels.append(x)

        img = np.stack(channels, axis=2)   # shape: (H, W, C)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)  # normalize to [0, 1]

        # Load segmentation mask
        seg = np.load(mask_path).astype(np.uint8)
        if seg.ndim == 3:
            seg = seg.squeeze()

        seg = cv2.resize(
            seg,
            (self.image_size, self.image_size),
            interpolation=cv2.INTER_NEAREST
        )
        seg = (seg > 0).astype(np.uint8)

        # Apply augmentation / center crop
        if self.center_crop:
            out = self.aug(image=img, mask=seg)
            img, seg = out["image"], out["mask"]

        # Convert to torch tensor
        img = torch.from_numpy(img.transpose(2, 0, 1))   # (C, H, W)
        seg = torch.from_numpy(seg)

        return img, seg, 0
