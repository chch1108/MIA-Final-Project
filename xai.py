
"""
Official-Mask Tumor Segmentation + Explainable AI Pipeline
=========================================================

Purpose
-------
This script is designed to replace the earlier pseudo-label/XAI workflow with a
ground-truth supervised workflow:

1. Use only BraTS2020 volumes whose H5 slices contain an official tumor mask.
2. Exclude positive-but-unannotated cases because they do not provide a valid IoU
   ground truth.
3. Train segmentation models on the official mask:
   - Proposed CrossModal-Swin Segmentation model
   - Attention 3D U-Net explainable model
4. Evaluate Dice / IoU on validation data.
5. Generate Grad-CAM++ / overlay / contour comparison figures:
   - Yellow contour = official tumor annotation
   - Cyan contour   = model prediction
   - Heatmap        = Grad-CAM++ activation
6. Export CSV summaries and figures.

Typical usage
-------------
Quick test:
    python train_official_mask_segmentation_xai.py --download-kaggle --epochs 2 --max-volumes 16

Recommended first run:
    python train_official_mask_segmentation_xai.py --download-kaggle --epochs 10 --num-xai-samples 4

Manual dataset root:
    python train_official_mask_segmentation_xai.py --dataset-root "D:\\path\\to\\dataset" --epochs 10

Notes
-----
- This script does NOT use pseudo labels.
- This script excludes any volume without an explicit H5 mask.
- By default it trains binary tumor segmentation:
      background vs tumor
  This is the most direct setup for improving tumor-region IoU.
- If you later want NCR/NET, ED, ET multi-class segmentation, set:
      --task multiclass
  but first confirm that the H5 mask class values are valid.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Sequence

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


# =============================================================================
# 0. Utilities
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def save_json(obj, path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def download_kaggle_dataset() -> Path:
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError("Please install kagglehub first: pip install kagglehub") from exc

    print("Downloading / reusing Kaggle BraTS2020 training data cache...")
    path = kagglehub.dataset_download("awsaf49/brats2020-training-data")
    print(f"Dataset root: {path}")
    return Path(path)


def resolve_dataset_paths(dataset_root: Path) -> Dict[str, Path]:
    candidates_data_dir = [
        dataset_root / "BraTS2020_training_data" / "content" / "data",
        dataset_root / "content" / "data",
        dataset_root / "data",
        dataset_root,
    ]
    data_dir = next((p for p in candidates_data_dir if p.exists() and any(p.glob("volume_*_slice_*.h5"))), None)
    if data_dir is None:
        raise FileNotFoundError(
            "Could not find H5 data folder containing files like volume_1_slice_0.h5. "
            "Check --dataset-root."
        )

    meta_candidates = [
        dataset_root / "BraTS20 Training Metadata.csv",
        data_dir / "BraTS20 Training Metadata.csv",
        dataset_root / "meta_data.csv",
        data_dir / "meta_data.csv",
    ]
    meta_csv = next((p for p in meta_candidates if p.exists()), None)

    if meta_csv is None:
        # Build minimal metadata by scanning file names.
        print("[Warning] Metadata CSV not found. Building metadata by scanning H5 file names.")
        rows = []
        for fp in sorted(data_dir.glob("volume_*_slice_*.h5")):
            stem = fp.stem
            # volume_1_slice_0
            parts = stem.split("_")
            if len(parts) >= 4:
                rows.append({"volume": int(parts[1]), "slice": int(parts[3]), "path": str(fp)})
        if not rows:
            raise FileNotFoundError("No H5 files found.")
        generated_meta = data_dir / "_generated_metadata.csv"
        pd.DataFrame(rows).to_csv(generated_meta, index=False)
        meta_csv = generated_meta

    return {
        "dataset_root": dataset_root,
        "data_dir": data_dir,
        "meta_csv": meta_csv,
    }


def get_h5_keys(fp: Path) -> List[str]:
    with h5py.File(fp, "r") as f:
        return list(f.keys())


def read_h5_image_mask(fp: Path, mask_key_candidates: Sequence[str] = ("mask", "seg", "label", "labels", "segmentation")):
    with h5py.File(fp, "r") as f:
        if "image" not in f:
            raise KeyError(f"H5 file has no 'image' key: {fp}. Keys={list(f.keys())}")
        image = f["image"][()].astype(np.float32)  # expected (H, W, 4)

        mask = None
        used_key = None
        for k in mask_key_candidates:
            if k in f:
                mask = f[k][()]
                used_key = k
                break
        if mask is None:
            return image, None, None

        mask = np.asarray(mask)
        # Common formats: (H,W), (H,W,1), one-hot (H,W,C)
        if mask.ndim == 3:
            if mask.shape[-1] == 1:
                mask = mask[..., 0]
            elif mask.shape[-1] > 1:
                # If one-hot probability/class channels, convert to class id.
                mask = np.argmax(mask, axis=-1)
            else:
                mask = np.squeeze(mask)
        mask = mask.astype(np.int16)
        return image, mask, used_key


def normalize_modalities(chw: np.ndarray) -> np.ndarray:
    # chw: (4,H,W)
    out = chw.astype(np.float32).copy()
    for c in range(out.shape[0]):
        mask = out[c] > 0
        if mask.sum() > 0:
            mu = out[c][mask].mean()
            sd = out[c][mask].std()
            out[c] = np.where(mask, (out[c] - mu) / (sd + 1e-8), 0.0)
    return np.clip(out, -5, 5)


def remap_mask(mask: np.ndarray, task: str) -> np.ndarray:
    """
    BraTS-style labels often include:
        0 background, 1 NCR/NET, 2 ED, 4 ET
    This function maps them to:
        binary: 0 background, 1 tumor
        multiclass: 0 background, 1 NCR/NET, 2 ED, 3 ET
    """
    mask = np.asarray(mask)
    if task == "binary":
        return (mask > 0).astype(np.int64)

    out = np.zeros_like(mask, dtype=np.int64)
    out[mask == 1] = 1
    out[mask == 2] = 2
    out[mask == 4] = 3
    # If a dataset already uses 3 for ET, preserve it.
    out[mask == 3] = 3
    return out.astype(np.int64)


def crop_or_pad_3d(img: np.ndarray, mask: np.ndarray, start_d: int, start_h: int, start_w: int, crop_d: int, crop_hw: int):
    """
    img  : (4,D,H,W)
    mask : (D,H,W)
    Return cropped/padded img and mask.
    """
    C, D, H, W = img.shape
    out_img = np.zeros((C, crop_d, crop_hw, crop_hw), dtype=np.float32)
    out_mask = np.zeros((crop_d, crop_hw, crop_hw), dtype=np.int64)

    d1, h1, w1 = start_d + crop_d, start_h + crop_hw, start_w + crop_hw

    src_d0 = max(0, start_d)
    src_h0 = max(0, start_h)
    src_w0 = max(0, start_w)
    src_d1 = min(D, d1)
    src_h1 = min(H, h1)
    src_w1 = min(W, w1)

    dst_d0 = src_d0 - start_d
    dst_h0 = src_h0 - start_h
    dst_w0 = src_w0 - start_w
    dst_d1 = dst_d0 + (src_d1 - src_d0)
    dst_h1 = dst_h0 + (src_h1 - src_h0)
    dst_w1 = dst_w0 + (src_w1 - src_w0)

    out_img[:, dst_d0:dst_d1, dst_h0:dst_h1, dst_w0:dst_w1] = img[:, src_d0:src_d1, src_h0:src_h1, src_w0:src_w1]
    out_mask[dst_d0:dst_d1, dst_h0:dst_h1, dst_w0:dst_w1] = mask[src_d0:src_d1, src_h0:src_h1, src_w0:src_w1]
    return out_img, out_mask


def bbox_from_mask(mask: np.ndarray) -> Optional[Tuple[int, int, int, int, int, int]]:
    coords = np.argwhere(mask > 0)
    if coords.size == 0:
        return None
    d0, h0, w0 = coords.min(axis=0)
    d1, h1, w1 = coords.max(axis=0) + 1
    return int(d0), int(d1), int(h0), int(h1), int(w0), int(w1)


def choose_crop_start(mask: np.ndarray, crop_d: int, crop_hw: int, train: bool, jitter: int = 12):
    D, H, W = mask.shape
    bbox = bbox_from_mask(mask)
    if bbox is None:
        center_d, center_h, center_w = D // 2, H // 2, W // 2
    else:
        d0, d1, h0, h1, w0, w1 = bbox
        center_d = (d0 + d1) // 2
        center_h = (h0 + h1) // 2
        center_w = (w0 + w1) // 2
        if train:
            center_d += random.randint(-jitter, jitter)
            center_h += random.randint(-jitter, jitter)
            center_w += random.randint(-jitter, jitter)

    start_d = int(center_d - crop_d // 2)
    start_h = int(center_h - crop_hw // 2)
    start_w = int(center_w - crop_hw // 2)

    start_d = min(max(start_d, 0), max(0, D - crop_d))
    start_h = min(max(start_h, 0), max(0, H - crop_hw))
    start_w = min(max(start_w, 0), max(0, W - crop_hw))
    return start_d, start_h, start_w


def find_best_tumor_slice(mask: np.ndarray) -> int:
    counts = (mask > 0).sum(axis=(1, 2))
    return int(np.argmax(counts))


def normalize_2d(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x


def draw_contours_rgb(base_rgb: np.ndarray, mask: np.ndarray, color=(1.0, 1.0, 0.0), thickness: int = 2) -> np.ndarray:
    """
    Draw contours on RGB image in [0,1].
    color default: yellow.
    """
    out = np.clip(base_rgb.copy(), 0, 1)
    binary = (mask > 0).astype(np.uint8)
    if binary.sum() == 0:
        return out

    if HAS_CV2:
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img_u8 = np.uint8(out * 255)
        bgr = tuple(int(c * 255) for c in color[::-1])
        cv2.drawContours(img_u8, contours, -1, bgr, thickness)
        return img_u8.astype(np.float32) / 255.0

    # Fallback: simple boundary by erosion difference.
    import scipy.ndimage as ndi
    eroded = ndi.binary_erosion(binary, iterations=1)
    boundary = binary.astype(bool) & ~eroded.astype(bool)
    for c in range(3):
        out[..., c][boundary] = color[c]
    return out


def overlay_mask_red(base_rgb: np.ndarray, mask: np.ndarray, alpha: float = 0.35) -> np.ndarray:
    out = base_rgb.copy()
    m = mask > 0
    red = np.zeros_like(out)
    red[..., 0] = 1.0
    out[m] = (1 - alpha) * out[m] + alpha * red[m]
    return np.clip(out, 0, 1)


def overlay_heatmap(base_slice: np.ndarray, heatmap_slice: np.ndarray, alpha: float = 0.40) -> np.ndarray:
    base = normalize_2d(base_slice)
    base_rgb = np.stack([base] * 3, axis=-1)
    hm = np.clip(heatmap_slice, 0, 1).astype(np.float32)

    if HAS_CV2:
        color = cv2.applyColorMap(np.uint8(hm * 255), cv2.COLORMAP_JET)
        color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    else:
        color = plt.get_cmap("jet")(hm)[..., :3].astype(np.float32)

    return np.clip((1 - alpha) * base_rgb + alpha * color, 0, 1)


# =============================================================================
# 1. Scan Official-Mask Volumes
# =============================================================================
@dataclass
class VolumeRecord:
    volume_id: int
    n_slices: int
    tumor_voxels: int
    mask_key: str
    best_slice: int


def scan_official_mask_volumes(paths: Dict[str, Path], task: str, min_tumor_voxels: int, max_volumes: int = 0) -> List[VolumeRecord]:
    df_meta = pd.read_csv(paths["meta_csv"])
    if "volume" not in df_meta.columns or "slice" not in df_meta.columns:
        raise ValueError(f"Metadata CSV must contain 'volume' and 'slice' columns. Columns={list(df_meta.columns)}")

    data_dir = paths["data_dir"]
    records: List[VolumeRecord] = []

    volume_ids = sorted([int(v) for v in df_meta["volume"].unique()])
    if max_volumes and max_volumes > 0:
        volume_ids = volume_ids[:max_volumes]

    print("Scanning volumes for explicit official H5 masks...")
    for vid in tqdm(volume_ids, desc="Scan official masks"):
        vol_df = df_meta[df_meta["volume"] == vid].sort_values("slice")
        tumor_counts = []
        used_key = None
        ok = True

        for _, row in vol_df.iterrows():
            sl = int(row["slice"])
            fp = data_dir / f"volume_{vid}_slice_{sl}.h5"
            if not fp.exists():
                ok = False
                break
            try:
                _, mask, key = read_h5_image_mask(fp)
            except Exception:
                ok = False
                break
            if mask is None or key is None:
                ok = False
                break
            used_key = key
            mapped = remap_mask(mask, task=task)
            tumor_counts.append(int((mapped > 0).sum()))

        if not ok or used_key is None:
            continue

        total_tumor = int(np.sum(tumor_counts))
        if total_tumor >= min_tumor_voxels:
            best_slice = int(np.argmax(tumor_counts))
            records.append(
                VolumeRecord(
                    volume_id=vid,
                    n_slices=len(tumor_counts),
                    tumor_voxels=total_tumor,
                    mask_key=used_key,
                    best_slice=best_slice,
                )
            )

    print(f"Official-mask positive volumes: {len(records)}")
    if records:
        print("Examples:", [asdict(r) for r in records[:5]])
    return records


# =============================================================================
# 2. Dataset
# =============================================================================
class OfficialMaskBraTSDataset(Dataset):
    def __init__(
        self,
        records: List[VolumeRecord],
        paths: Dict[str, Path],
        task: str = "binary",
        crop_d: int = 64,
        crop_hw: int = 128,
        train: bool = True,
        jitter: int = 12,
    ):
        self.records = records
        self.paths = paths
        self.task = task
        self.crop_d = crop_d
        self.crop_hw = crop_hw
        self.train = train
        self.jitter = jitter
        self.df_meta = pd.read_csv(paths["meta_csv"])

    def __len__(self) -> int:
        return len(self.records)

    def load_full_volume(self, volume_id: int) -> Tuple[np.ndarray, np.ndarray]:
        vol_df = self.df_meta[self.df_meta["volume"] == volume_id].sort_values("slice")
        img_stack = []
        mask_stack = []
        for _, row in vol_df.iterrows():
            sl = int(row["slice"])
            fp = self.paths["data_dir"] / f"volume_{volume_id}_slice_{sl}.h5"
            image, mask, key = read_h5_image_mask(fp)
            if mask is None:
                raise RuntimeError(f"Volume {volume_id} slice {sl} has no official mask.")

            # image (H,W,4) -> (4,H,W), normalize per slice/channel
            chw = image.transpose(2, 0, 1)
            chw = normalize_modalities(chw)
            img_stack.append(chw)
            mask_stack.append(remap_mask(mask, task=self.task))

        img = np.stack(img_stack, axis=1).astype(np.float32)      # (4,D,H,W)
        mask = np.stack(mask_stack, axis=0).astype(np.int64)      # (D,H,W)
        return img, mask

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        img, mask = self.load_full_volume(rec.volume_id)

        sd, sh, sw = choose_crop_start(mask, self.crop_d, self.crop_hw, train=self.train, jitter=self.jitter)
        img_c, mask_c = crop_or_pad_3d(img, mask, sd, sh, sw, self.crop_d, self.crop_hw)

        img_t = torch.from_numpy(img_c).float()
        mask_t = torch.from_numpy(mask_c).long()

        meta = {
            "volume_id": rec.volume_id,
            "tumor_voxels_original": rec.tumor_voxels,
            "mask_key": rec.mask_key,
            "crop_start_d": sd,
            "crop_start_h": sh,
            "crop_start_w": sw,
        }
        return img_t, mask_t, meta


# =============================================================================
# 3. Models
# =============================================================================
class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(2, 8, 8), in_chans=4, embed_dim=48):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, E, Dp, Hp, Wp = x.shape
        tokens = x.flatten(2).transpose(1, 2)
        return self.norm(tokens), (Dp, Hp, Wp)


def window_partition3d(x, ws):
    B, D, H, W, C = x.shape
    x = x.view(B, D // ws[0], ws[0], H // ws[1], ws[1], W // ws[2], ws[2], C)
    x = x.permute(0, 1, 3, 5, 2, 4, 6, 7).contiguous()
    return x.view(-1, ws[0] * ws[1] * ws[2], C)


def window_reverse3d(windows, ws, D, H, W):
    B = int(windows.shape[0] / (D * H * W / (ws[0] * ws[1] * ws[2])))
    x = windows.view(B, D // ws[0], H // ws[1], W // ws[2], ws[0], ws[1], ws[2], -1)
    x = x.permute(0, 1, 4, 2, 5, 3, 6, 7).contiguous()
    return x.view(B, D, H, W, -1)


class WindowAttention3D(nn.Module):
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)


class SwinBlock3D(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=(2, 7, 7), mlp_ratio=4.0, shift=False):
        super().__init__()
        self.ws = window_size
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim * mlp_ratio), dim),
        )

    def forward(self, x, D, H, W):
        B, L, C = x.shape
        shortcut = x
        x = self.norm1(x).view(B, D, H, W, C)

        pd = (self.ws[0] - D % self.ws[0]) % self.ws[0]
        ph = (self.ws[1] - H % self.ws[1]) % self.ws[1]
        pw = (self.ws[2] - W % self.ws[2]) % self.ws[2]
        if pd or ph or pw:
            x = F.pad(x, (0, 0, 0, pw, 0, ph, 0, pd))
        _, Dp, Hp, Wp, _ = x.shape

        if self.shift:
            shift = (self.ws[0] // 2, self.ws[1] // 2, self.ws[2] // 2)
            x = torch.roll(x, shifts=(-shift[0], -shift[1], -shift[2]), dims=(1, 2, 3))

        windows = window_partition3d(x, self.ws)
        attn_out = self.attn(windows)
        x = window_reverse3d(attn_out, self.ws, Dp, Hp, Wp)

        if self.shift:
            x = torch.roll(x, shifts=shift, dims=(1, 2, 3))

        x = x[:, :D, :H, :W, :].contiguous().view(B, L, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class ModalityEncoder(nn.Module):
    def __init__(self, in_chans=2, embed_dim=48, patch_size=(2, 8, 8), depth=2, num_heads=4, window_size=(2, 7, 7)):
        super().__init__()
        self.patch_embed = PatchEmbed3D(patch_size, in_chans, embed_dim)
        self.blocks = nn.ModuleList([
            SwinBlock3D(embed_dim, num_heads, window_size=window_size, shift=(i % 2 == 1))
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        feat, grid = self.patch_embed(x)
        D, H, W = grid
        for blk in self.blocks:
            feat = blk(feat, D, H, W)
        return self.norm(feat), grid


class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.nq = nn.LayerNorm(dim)
        self.nkv = nn.LayerNorm(dim)

    def forward(self, feat_a, feat_b):
        B, L, C = feat_a.shape
        H = self.num_heads
        Hd = C // H
        q = self.q(self.nq(feat_a)).reshape(B, L, H, Hd).transpose(1, 2)
        k = self.k(self.nkv(feat_b)).reshape(B, L, H, Hd).transpose(1, 2)
        v = self.v(feat_b).reshape(B, L, H, Hd).transpose(1, 2)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, L, C)
        return self.out(x)


class CrossModalSwinSeg(nn.Module):
    """
    Segmentation-only version of the earlier CrossModal-Swin idea.
    Input:  4-channel MRI crop (T1, T1Gd, T2, FLAIR)
    Output: segmentation logits (B, num_classes, D, H, W)
    """
    def __init__(self, num_classes=2, crop_d=64, crop_hw=128, embed_dim=48, depth=2, num_heads=4):
        super().__init__()
        patch_size = (2, 8, 8)
        window_size = (2, 7, 7)
        self.enc_t1gd = ModalityEncoder(2, embed_dim, patch_size, depth, num_heads, window_size)
        self.enc_t2flair = ModalityEncoder(2, embed_dim, patch_size, depth, num_heads, window_size)
        self.cross_attn = CrossModalAttention(embed_dim, num_heads)
        self.fusion_norm = nn.LayerNorm(embed_dim * 2)
        self.seg_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim * 2),
            nn.GELU(),
            nn.Linear(embed_dim * 2, num_classes),
        )
        self.crop_d = crop_d
        self.crop_hw = crop_hw

    def forward(self, x):
        # T1/T1Gd and T2/FLAIR branches.
        f1, grid = self.enc_t1gd(x[:, :2])
        f2, _ = self.enc_t2flair(x[:, 2:])
        cross = self.cross_attn(f1, f2)
        fused = self.fusion_norm(torch.cat([cross, f2], dim=-1))
        D, H, W = grid
        B = x.shape[0]
        logits = self.seg_head(fused).transpose(1, 2).view(B, -1, D, H, W)
        logits = F.interpolate(logits.float(), size=(self.crop_d, self.crop_hw, self.crop_hw), mode="trilinear", align_corners=False)
        return logits


class ConvBlock3D(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class AttentionGate3D(nn.Module):
    def __init__(self, g_c, x_c, inter_c):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv3d(g_c, inter_c, 1), nn.InstanceNorm3d(inter_c))
        self.W_x = nn.Sequential(nn.Conv3d(x_c, inter_c, 1), nn.InstanceNorm3d(inter_c))
        self.psi = nn.Sequential(nn.Conv3d(inter_c, 1, 1), nn.Sigmoid())
        self.relu = nn.ReLU(inplace=True)
        self.last_attention: Optional[torch.Tensor] = None

    def forward(self, g, x):
        a = self.relu(self.W_g(g) + self.W_x(x))
        psi = self.psi(a)
        self.last_attention = psi.detach()
        return x * psi


class AttentionUNet3D(nn.Module):
    """
    Explainable 3D U-Net with Attention Gates.
    Attention maps can be visualized in addition to Grad-CAM++.
    """
    def __init__(self, in_ch=4, num_classes=2, base_c=16):
        super().__init__()
        c = base_c
        self.enc1 = ConvBlock3D(in_ch, c)
        self.enc2 = ConvBlock3D(c, c * 2)
        self.enc3 = ConvBlock3D(c * 2, c * 4)
        self.bot = ConvBlock3D(c * 4, c * 8)
        self.pool = nn.MaxPool3d(2)

        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, 2, stride=2)
        self.att3 = AttentionGate3D(c * 4, c * 4, c * 2)
        self.dec3 = ConvBlock3D(c * 8, c * 4)

        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, 2, stride=2)
        self.att2 = AttentionGate3D(c * 2, c * 2, c)
        self.dec2 = ConvBlock3D(c * 4, c * 2)

        self.up1 = nn.ConvTranspose3d(c * 2, c, 2, stride=2)
        self.att1 = AttentionGate3D(c, c, max(1, c // 2))
        self.dec1 = ConvBlock3D(c * 2, c)

        self.out = nn.Conv3d(c, num_classes, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bot(self.pool(e3))

        u3 = self.up3(b)
        a3 = self.att3(u3, e3)
        d3 = self.dec3(torch.cat([u3, a3], dim=1))

        u2 = self.up2(d3)
        a2 = self.att2(u2, e2)
        d2 = self.dec2(torch.cat([u2, a2], dim=1))

        u1 = self.up1(d2)
        a1 = self.att1(u1, e1)
        d1 = self.dec1(torch.cat([u1, a1], dim=1))

        return self.out(d1)


# =============================================================================
# 4. Loss and metrics
# =============================================================================
class DiceLossMulticlass(nn.Module):
    def __init__(self, smooth=1e-5, ignore_bg=True):
        super().__init__()
        self.smooth = smooth
        self.ignore_bg = ignore_bg

    def forward(self, logits, target):
        num_classes = logits.shape[1]
        probs = torch.softmax(logits, dim=1)
        target_oh = F.one_hot(target.clamp_min(0), num_classes).permute(0, 4, 1, 2, 3).float()
        start = 1 if self.ignore_bg and num_classes > 1 else 0
        losses = []
        for c in range(start, num_classes):
            p = probs[:, c]
            g = target_oh[:, c]
            inter = (p * g).sum(dim=(1, 2, 3))
            denom = p.sum(dim=(1, 2, 3)) + g.sum(dim=(1, 2, 3))
            dice = (2 * inter + self.smooth) / (denom + self.smooth)
            losses.append(1 - dice)
        return torch.stack(losses, dim=0).mean()


def segmentation_loss(logits, target, dice_weight=0.6, ce_weight=0.4):
    dice = DiceLossMulticlass(ignore_bg=True)(logits, target)
    ce = F.cross_entropy(logits, target)
    return dice_weight * dice + ce_weight * ce


def compute_metrics_from_logits(logits: torch.Tensor, target: torch.Tensor, num_classes: int) -> Dict[str, float]:
    pred = logits.argmax(dim=1)
    metrics = {}

    # binary tumor metric: any non-background.
    pred_bin = pred > 0
    gt_bin = target > 0
    inter = torch.logical_and(pred_bin, gt_bin).sum().item()
    union = torch.logical_or(pred_bin, gt_bin).sum().item()
    pred_sum = pred_bin.sum().item()
    gt_sum = gt_bin.sum().item()
    metrics["tumor_iou"] = float(inter / union) if union > 0 else float("nan")
    metrics["tumor_dice"] = float((2 * inter) / (pred_sum + gt_sum)) if (pred_sum + gt_sum) > 0 else float("nan")

    # per-class tumor metrics.
    names = {1: "ncr_net", 2: "ed", 3: "et"}
    class_ious = []
    class_dices = []
    for cls in range(1, num_classes):
        p = pred == cls
        g = target == cls
        inter_c = torch.logical_and(p, g).sum().item()
        union_c = torch.logical_or(p, g).sum().item()
        ps = p.sum().item()
        gs = g.sum().item()
        iou = float(inter_c / union_c) if union_c > 0 else float("nan")
        dice = float((2 * inter_c) / (ps + gs)) if (ps + gs) > 0 else float("nan")
        key = names.get(cls, f"class{cls}")
        metrics[f"{key}_iou"] = iou
        metrics[f"{key}_dice"] = dice
        if not math.isnan(iou):
            class_ious.append(iou)
        if not math.isnan(dice):
            class_dices.append(dice)

    metrics["mean_class_iou"] = float(np.mean(class_ious)) if class_ious else float("nan")
    metrics["mean_class_dice"] = float(np.mean(class_dices)) if class_dices else float("nan")
    return metrics


def average_metric_dict(rows: List[Dict[str, float]]) -> Dict[str, float]:
    keys = sorted({k for r in rows for k in r.keys()})
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r and not (isinstance(r[k], float) and math.isnan(r[k]))]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out


# =============================================================================
# 5. Train / Validate
# =============================================================================
def train_one_epoch(model, loader, optimizer, device, num_classes: int, use_amp: bool):
    model.train()
    total_loss = 0.0
    scaler = train_one_epoch.scaler if use_amp else None

    for img, mask, meta in tqdm(loader, desc="Train", leave=False):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(img)
                loss = segmentation_loss(logits, mask)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(img)
            loss = segmentation_loss(logits, mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        total_loss += float(loss.detach().cpu())

    return total_loss / max(1, len(loader))


train_one_epoch.scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None


@torch.no_grad()
def validate_model(model, loader, device, num_classes: int) -> Dict[str, float]:
    model.eval()
    losses = []
    rows = []
    for img, mask, meta in tqdm(loader, desc="Val", leave=False):
        img = img.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        logits = model(img)
        loss = segmentation_loss(logits, mask)
        losses.append(float(loss.detach().cpu()))
        rows.append(compute_metrics_from_logits(logits.detach().cpu(), mask.detach().cpu(), num_classes=num_classes))

    metrics = average_metric_dict(rows)
    metrics["val_loss"] = float(np.mean(losses)) if losses else float("nan")
    return metrics


def train_model(
    model: nn.Module,
    model_name: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device,
    output_dir: Path,
    epochs: int,
    lr: float,
    weight_decay: float,
    num_classes: int,
    use_amp: bool,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    history = {
        "train_loss": [],
        "val_loss": [],
        "tumor_iou": [],
        "tumor_dice": [],
        "mean_class_iou": [],
        "mean_class_dice": [],
    }

    best_score = -1.0
    best_path = output_dir / f"best_{model_name}.pth"
    last_path = output_dir / f"last_{model_name}.pth"

    print(f"\n=== Training {model_name} for {epochs} epochs ===")
    for ep in range(1, epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, device, num_classes=num_classes, use_amp=use_amp)
        val = validate_model(model, val_loader, device, num_classes=num_classes)
        scheduler.step()

        history["train_loss"].append(tr_loss)
        for k in ["val_loss", "tumor_iou", "tumor_dice", "mean_class_iou", "mean_class_dice"]:
            history[k].append(val.get(k, float("nan")))

        score = val.get("tumor_dice", float("nan"))
        if not math.isnan(score) and score > best_score:
            best_score = score
            torch.save(model.state_dict(), best_path)
            flag = " ⭐ best"
        else:
            flag = ""

        print(
            f"Epoch {ep:02d}/{epochs} | "
            f"TrainLoss={tr_loss:.4f} | ValLoss={val['val_loss']:.4f} | "
            f"Tumor IoU={val.get('tumor_iou', float('nan')):.4f} | "
            f"Tumor Dice={val.get('tumor_dice', float('nan')):.4f}{flag}"
        )

    torch.save(model.state_dict(), last_path)
    save_json(history, output_dir / f"history_{model_name}.json")
    return history, best_path


# =============================================================================
# 6. Grad-CAM++
# =============================================================================
class GradCAMPlusPlus3D:
    """
    Generic Grad-CAM++ for 3D feature maps.
    Works with Conv3D target layers where activation shape is (B,C,D,H,W).

    For token-based Swin layer, use SwinTokenGradCAMPlusPlus below.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.fh = target_layer.register_forward_hook(self._forward_hook)
        self.bh = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.fh.remove()
        self.bh.remove()

    def compute(self, img: torch.Tensor, class_idx: int = 1):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(img)
        score = logits[:, class_idx].mean()
        score.backward(retain_graph=False)

        if self.activations is None or self.gradients is None:
            raise RuntimeError("GradCAM hooks did not capture activations or gradients.")

        acts = self.activations.detach()[0]  # (C,D,H,W)
        grads = self.gradients.detach()[0]  # (C,D,H,W)

        grads2 = grads ** 2
        grads3 = grads2 * grads
        sum_acts = acts.sum(dim=(1, 2, 3), keepdim=True)
        alpha = grads2 / (2 * grads2 + sum_acts * grads3 + 1e-7)
        weights = (torch.relu(grads) * alpha).sum(dim=(1, 2, 3))
        cam = torch.relu((weights[:, None, None, None] * acts).sum(dim=0))[None, None]
        cam = F.interpolate(cam, size=img.shape[2:], mode="trilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy().astype(np.float32)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


class SwinTokenGradCAMPlusPlus:
    """
    Grad-CAM++ for token outputs shaped (B,L,C).
    It approximates Grad-CAM++ in token space and reshapes back to grid.
    """
    def __init__(self, model: nn.Module, target_layer: nn.Module, grid_shape: Tuple[int, int, int]):
        self.model = model
        self.target_layer = target_layer
        self.grid_shape = grid_shape
        self.activations = None
        self.gradients = None
        self.fh = target_layer.register_forward_hook(self._forward_hook)
        self.bh = target_layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, inputs, output):
        self.activations = output[0] if isinstance(output, tuple) else output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        self.fh.remove()
        self.bh.remove()

    def compute(self, img: torch.Tensor, class_idx: int = 1):
        self.model.eval()
        self.model.zero_grad(set_to_none=True)

        logits = self.model(img)
        score = logits[:, class_idx].mean()
        score.backward(retain_graph=False)

        acts = self.activations.detach()[0]  # (L,C)
        grads = self.gradients.detach()[0]   # (L,C)

        # Token Grad-CAM++ weighting.
        grads2 = grads ** 2
        grads3 = grads2 * grads
        sum_acts = acts.sum(dim=0, keepdim=True)  # (1,C)
        alpha = grads2 / (2 * grads2 + sum_acts * grads3 + 1e-7)
        weights = (torch.relu(grads) * alpha).sum(dim=0)  # (C,)

        cam_tokens = torch.relu((acts * weights).sum(dim=1))  # (L,)
        Dp, Hp, Wp = self.grid_shape
        if cam_tokens.numel() != Dp * Hp * Wp:
            raise RuntimeError(f"Token shape mismatch: {cam_tokens.numel()} vs {Dp*Hp*Wp}")

        cam = cam_tokens.reshape(1, 1, Dp, Hp, Wp)
        cam = F.interpolate(cam, size=img.shape[2:], mode="trilinear", align_corners=False)[0, 0]
        cam = cam.detach().cpu().numpy().astype(np.float32)
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam


@torch.no_grad()
def predict_mask(model, img: torch.Tensor, device) -> np.ndarray:
    model.eval()
    logits = model(img.to(device))
    pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int16)
    return pred


def attention_map_from_unet(model: AttentionUNet3D, target_shape: Tuple[int, int, int]) -> Optional[np.ndarray]:
    att = model.att1.last_attention
    if att is None:
        return None
    att = F.interpolate(att.float(), size=target_shape, mode="trilinear", align_corners=False)[0, 0]
    arr = att.detach().cpu().numpy().astype(np.float32)
    arr -= arr.min()
    if arr.max() > 0:
        arr /= arr.max()
    return arr


# =============================================================================
# 7. Visualization
# =============================================================================
def plot_training_curves(all_histories: Dict[str, Dict[str, list]], output_dir: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    panels = [
        ("val_loss", "Validation Loss"),
        ("tumor_iou", "Tumor IoU"),
        ("tumor_dice", "Tumor Dice"),
    ]
    for ax, (key, title) in zip(axes, panels):
        for name, hist in all_histories.items():
            vals = hist.get(key, [])
            ax.plot(np.arange(1, len(vals) + 1), vals, marker="o", label=name)
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    plt.tight_layout()
    fp = output_dir / "fig01_training_curves_iou_dice.png"
    plt.savefig(fp, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {fp}")


def create_xai_figure(
    models: Dict[str, nn.Module],
    best_paths: Dict[str, Path],
    val_dataset: OfficialMaskBraTSDataset,
    device,
    output_dir: Path,
    num_samples: int,
    num_classes: int,
    class_idx_for_cam: int,
    crop_d: int,
    crop_hw: int,
    embed_dim: int,
    patch_size: Tuple[int, int, int],
    seed: int,
):
    """
    Simplified XAI figure.

    Output only two columns:
        1. Official annotation
        2. Proposed-CrossModalSwin prediction vs official annotation

    This figure intentionally removes Grad-CAM, Attention-3D-UNet, and predicted-area
    panels from fig02 because the user only needs a clear contour comparison for
    official mask vs Proposed model prediction.
    """
    rng = random.Random(seed)
    indices = list(range(len(val_dataset)))
    rng.shuffle(indices)
    indices = indices[: min(num_samples, len(indices))]

    proposed_name = "Proposed-CrossModalSwin"
    if proposed_name not in models:
        raise ValueError(
            f"{proposed_name} is required for simplified fig02 output. "
            "Please do not run with --skip-swin."
        )

    proposed_model = models[proposed_name]
    proposed_model.eval()
    if best_paths.get(proposed_name) is not None and Path(best_paths[proposed_name]).exists():
        proposed_model.load_state_dict(torch.load(best_paths[proposed_name], map_location=device))
        proposed_model.to(device)
        proposed_model.eval()

    summary_rows = []
    n_rows = len(indices)
    n_cols = 2
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(8.2, 4.2 * n_rows))
    if n_rows == 1:
        axes = np.expand_dims(axes, axis=0)

    yellow = (1.0, 1.0, 0.0)
    cyan = (0.0, 1.0, 1.0)

    for row_i, ds_idx in enumerate(indices):
        img, mask, meta = val_dataset[ds_idx]
        img_b = img.unsqueeze(0).to(device)
        mask_np = mask.numpy().astype(np.int16)
        slice_idx = find_best_tumor_slice(mask_np)
        mri_slice = img[1, slice_idx].numpy()
        base_rgb = np.stack([normalize_2d(mri_slice)] * 3, axis=-1)
        official_binary_slice = mask_np[slice_idx] > 0

        vol_id = int(meta["volume_id"][0] if isinstance(meta["volume_id"], torch.Tensor) else meta["volume_id"])

        # Column 1: official annotation only.
        official_img = overlay_mask_red(base_rgb, official_binary_slice, alpha=0.35)
        official_img = draw_contours_rgb(official_img, official_binary_slice, color=yellow, thickness=2)
        axes[row_i, 0].imshow(official_img)
        axes[row_i, 0].set_title(
            f"Volume {vol_id}\nOfficial annotation | Slice {slice_idx}\nYellow = official tumor",
            fontsize=10,
        )
        axes[row_i, 0].axis("off")

        # Column 2: Proposed prediction vs official annotation.
        with torch.no_grad():
            logits = proposed_model(img_b)
            pred = logits.argmax(dim=1)[0].detach().cpu().numpy().astype(np.int16)

        metrics3d = compute_metrics_from_logits(logits.detach().cpu(), mask.unsqueeze(0), num_classes=num_classes)
        pred_bin_slice = pred[slice_idx] > 0

        inter_s = np.logical_and(pred_bin_slice, official_binary_slice).sum()
        union_s = np.logical_or(pred_bin_slice, official_binary_slice).sum()
        ps = pred_bin_slice.sum()
        gs = official_binary_slice.sum()
        slice_iou = float(inter_s / union_s) if union_s > 0 else float("nan")
        slice_dice = float(2 * inter_s / (ps + gs)) if (ps + gs) > 0 else float("nan")

        summary_rows.append({
            "Model": proposed_name,
            "Volume_ID": vol_id,
            "Slice_Index": int(slice_idx),
            "Tumor_IoU_3D": metrics3d.get("tumor_iou", float("nan")),
            "Tumor_Dice_3D": metrics3d.get("tumor_dice", float("nan")),
            "Tumor_IoU_Slice": slice_iou,
            "Tumor_Dice_Slice": slice_dice,
            "Official_Tumor_Pixels_Slice": int(gs),
            "Pred_Tumor_Pixels_Slice": int(ps),
        })

        pred_rgb = base_rgb.copy()
        pred_rgb = draw_contours_rgb(pred_rgb, official_binary_slice, color=yellow, thickness=2)
        pred_rgb = draw_contours_rgb(pred_rgb, pred_bin_slice, color=cyan, thickness=2)
        axes[row_i, 1].imshow(pred_rgb)
        axes[row_i, 1].set_title(
            f"Proposed-CrossModalSwin\nPrediction vs official\n"
            f"Yellow = GT, Cyan = Pred\n"
            f"3D IoU = {metrics3d.get('tumor_iou', float('nan')):.3f} | "
            f"Slice IoU = {slice_iou:.3f}",
            fontsize=9,
        )
        axes[row_i, 1].axis("off")

    fig.suptitle(
        "Official-Mask Tumor Segmentation Comparison\n"
        "Only positive cases with explicit H5 tumor masks are used for IoU.",
        fontsize=15,
        fontweight="bold",
        y=0.995,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    fig_path = output_dir / "fig02_official_mask_prediction_iou.png"
    plt.savefig(fig_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    # Keep the old filename as an alias, so existing workflows that open fig02 still work.
    alias_path = output_dir / "fig02_official_mask_gradcam_attention_iou.png"
    try:
        import shutil
        shutil.copyfile(fig_path, alias_path)
    except Exception:
        pass

    csv_path = output_dir / "official_mask_xai_iou_summary.csv"
    pd.DataFrame(summary_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    print(f"Saved simplified XAI figure: {fig_path}")
    print(f"Saved alias figure: {alias_path}")
    print(f"Saved XAI summary: {csv_path}")


# =============================================================================
# 8. Argument parser and main
# =============================================================================
def build_argparser():
    p = argparse.ArgumentParser(description="Train official-mask tumor segmentation and generate Grad-CAM++ explainability.")
    p.add_argument("--dataset-root", type=str, default="", help="BraTS2020 dataset root. Optional if --download-kaggle is used.")
    p.add_argument("--download-kaggle", action="store_true", help="Download/reuse Kaggle cached BraTS2020 dataset.")
    p.add_argument("--output-dir", type=str, default="official_mask_seg_xai_outputs")
    p.add_argument("--task", type=str, default="binary", choices=["binary", "multiclass"], help="Segmentation task.")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=0, help="Windows usually works best with 0.")
    p.add_argument("--crop-d", type=int, default=64)
    p.add_argument("--crop-hw", type=int, default=128)
    p.add_argument("--embed-dim", type=int, default=48)
    p.add_argument("--depth", type=int, default=2)
    p.add_argument("--num-heads", type=int, default=4)
    p.add_argument("--base-c", type=int, default=16, help="Base channels for Attention 3D U-Net.")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--unet-lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--val-ratio", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--min-tumor-voxels", type=int, default=50)
    p.add_argument("--scan-max-volumes", type=int, default=0, help="0 = scan all volumes; use smaller number for debugging.")
    p.add_argument("--max-train-volumes", type=int, default=0, help="0 = use all official-mask positive volumes.")
    p.add_argument("--num-xai-samples", type=int, default=4)
    p.add_argument("--class-idx-for-cam", type=int, default=1, help="1=tumor for binary; 1/2/3 for multiclass.")
    p.add_argument("--skip-swin", action="store_true", help="Only train Attention 3D U-Net.")
    p.add_argument("--skip-attunet", action="store_true", help="Only train Proposed CrossModal-Swin.")
    p.add_argument("--skip-training", action="store_true", help="Skip training and load existing best weights from output-dir.")
    p.add_argument("--skip-xai", action="store_true", help="Skip Grad-CAM++ XAI output.")
    return p


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    if args.download_kaggle:
        dataset_root = download_kaggle_dataset()
    else:
        if not args.dataset_root:
            raise ValueError("Please provide --dataset-root or use --download-kaggle.")
        dataset_root = Path(args.dataset_root)

    paths = resolve_dataset_paths(dataset_root)
    print(f"Data dir: {paths['data_dir']}")
    print(f"Metadata: {paths['meta_csv']}")

    num_classes = 2 if args.task == "binary" else 4
    records = scan_official_mask_volumes(
        paths,
        task=args.task,
        min_tumor_voxels=args.min_tumor_voxels,
        max_volumes=args.scan_max_volumes,
    )

    if len(records) < 4:
        raise RuntimeError(
            f"Only {len(records)} official-mask positive volumes found. "
            f"Cannot reliably train/evaluate. Check whether your H5 files contain 'mask'."
        )

    if args.max_train_volumes and args.max_train_volumes > 0:
        records = records[: args.max_train_volumes]

    # Split by volume to avoid slice/volume leakage.
    train_records, val_records = train_test_split(
        records,
        test_size=args.val_ratio,
        random_state=args.seed,
        shuffle=True,
    )

    # If validation has fewer XAI cases, keep at least num_xai_samples if possible.
    if len(val_records) < min(args.num_xai_samples, len(records) // 3):
        n_val = min(max(args.num_xai_samples, 2), len(records) - 2)
        val_records = records[:n_val]
        train_records = records[n_val:]

    print(f"Train volumes: {len(train_records)} | Val volumes: {len(val_records)}")
    save_json(
        {
            "args": vars(args),
            "num_classes": num_classes,
            "train_records": [asdict(r) for r in train_records],
            "val_records": [asdict(r) for r in val_records],
        },
        output_dir / "dataset_split_records.json",
    )

    train_ds = OfficialMaskBraTSDataset(
        train_records, paths, task=args.task, crop_d=args.crop_d, crop_hw=args.crop_hw, train=True
    )
    val_ds = OfficialMaskBraTSDataset(
        val_records, paths, task=args.task, crop_d=args.crop_d, crop_hw=args.crop_hw, train=False
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    use_amp = torch.cuda.is_available()
    histories = {}
    best_paths = {}
    models: Dict[str, nn.Module] = {}

    if not args.skip_swin:
        swin = CrossModalSwinSeg(
            num_classes=num_classes,
            crop_d=args.crop_d,
            crop_hw=args.crop_hw,
            embed_dim=args.embed_dim,
            depth=args.depth,
            num_heads=args.num_heads,
        ).to(device)
        models["Proposed-CrossModalSwin"] = swin

    if not args.skip_attunet:
        attunet = AttentionUNet3D(
            in_ch=4,
            num_classes=num_classes,
            base_c=args.base_c,
        ).to(device)
        models["Attention-3D-UNet"] = attunet

    if not models:
        raise ValueError("No model selected. Do not set both --skip-swin and --skip-attunet.")

    if not args.skip_training:
        for name, model in models.items():
            lr = args.unet_lr if name == "Attention-3D-UNet" else args.lr
            hist, best_path = train_model(
                model=model,
                model_name=name.replace(" ", "_").replace("/", "_").replace("-", "_"),
                train_loader=train_loader,
                val_loader=val_loader,
                device=device,
                output_dir=output_dir,
                epochs=args.epochs,
                lr=lr,
                weight_decay=args.weight_decay,
                num_classes=num_classes,
                use_amp=use_amp,
            )
            histories[name] = hist
            best_paths[name] = best_path
        save_json(histories, output_dir / "all_histories.json")
        plot_training_curves(histories, output_dir)
    else:
        for name in models:
            fn = f"best_{name.replace(' ', '_').replace('/', '_').replace('-', '_')}.pth"
            pth = output_dir / fn
            if not pth.exists():
                raise FileNotFoundError(f"--skip-training requested but weight not found: {pth}")
            best_paths[name] = pth
            models[name].load_state_dict(torch.load(pth, map_location=device))
            models[name].to(device).eval()

    # Final validation metrics table.
    val_rows = []
    for name, model in models.items():
        if best_paths.get(name) and Path(best_paths[name]).exists():
            model.load_state_dict(torch.load(best_paths[name], map_location=device))
            model.to(device)
        val_metrics = validate_model(model, val_loader, device, num_classes=num_classes)
        row = {"Model": name, **val_metrics}
        val_rows.append(row)
        print(f"[Final Val] {name}: Tumor IoU={val_metrics.get('tumor_iou', float('nan')):.4f}, Tumor Dice={val_metrics.get('tumor_dice', float('nan')):.4f}")
    pd.DataFrame(val_rows).to_csv(output_dir / "final_validation_metrics.csv", index=False, encoding="utf-8-sig")

    if not args.skip_xai:
        create_xai_figure(
            models=models,
            best_paths=best_paths,
            val_dataset=val_ds,
            device=device,
            output_dir=output_dir,
            num_samples=args.num_xai_samples,
            num_classes=num_classes,
            class_idx_for_cam=args.class_idx_for_cam,
            crop_d=args.crop_d,
            crop_hw=args.crop_hw,
            embed_dim=args.embed_dim,
            patch_size=(2, 8, 8),
            seed=args.seed,
        )

    print("\n" + "=" * 80)
    print("Completed official-mask segmentation + XAI pipeline.")
    print(f"Output directory: {output_dir.resolve()}")
    print("Key outputs:")
    print("  - final_validation_metrics.csv")
    print("  - fig01_training_curves_iou_dice.png")
    print("  - fig02_official_mask_gradcam_attention_iou.png")
    print("  - official_mask_xai_iou_summary.csv")
    print("=" * 80)


if __name__ == "__main__":
    main()
