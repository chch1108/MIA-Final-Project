"""
BraTS20 Cross-Modal Swin-Transformer 腦瘤分析系統 — VS Code 版本

原始 Notebook: MIA_Final_2_0.ipynb
功能：
1. 下載/讀取 BraTS2020 training data
2. 讀取 4 模態 MRI: T1 / T1Gd / T2 / FLAIR
3. Cross-Modal Swin-Transformer 3D 多任務模型
4. 腫瘤分割 pseudo label 訓練 + 存活天數回歸 + 優先級分類
5. Baseline 比較與圖表輸出

注意：
- 原 notebook 內的分割標籤是「由影像強度規則產生的 pseudo label」，不是 BraTS 官方真實 segmentation label。
- 真實論文/正式實驗若要做 segmentation，請改成讀取資料集中真實 segmentation mask。
- 3D 影像很吃 GPU 記憶體；若電腦記憶體不足，請調小 --crop-d、--crop-hw、--embed-dim 或使用 --max-volumes 測試。
"""

from __future__ import annotations

import argparse
import json
import os
import random
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import h5py
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

warnings.filterwarnings("ignore")


# =============================================================================
# 0. 工具函式
# =============================================================================
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def download_kaggle_dataset() -> Path:
    """使用 kagglehub 下載資料集，需先準備 Kaggle API 憑證。"""
    try:
        import kagglehub
    except ImportError as exc:
        raise ImportError("請先安裝 kagglehub：pip install kagglehub") from exc

    print("開始下載 Kaggle BraTS2020 training data...")
    path = kagglehub.dataset_download("awsaf49/brats2020-training-data")
    print(f"資料集已下載至：{path}")
    return Path(path)


def resolve_dataset_paths(dataset_root: Path) -> Dict[str, Path]:
    """自動解析 notebook 中使用的資料路徑。"""
    candidates_data_dir = [
        dataset_root / "BraTS2020_training_data" / "content" / "data",
        dataset_root / "content" / "data",
        dataset_root / "data",
    ]
    data_dir = next((p for p in candidates_data_dir if p.exists()), None)
    if data_dir is None:
        raise FileNotFoundError(
            "找不到資料資料夾。請確認 dataset_root 底下是否有 "
            "BraTS2020_training_data/content/data。"
        )

    meta_candidates = [
        dataset_root / "BraTS20 Training Metadata.csv",
        data_dir / "BraTS20 Training Metadata.csv",
    ]
    meta_csv = next((p for p in meta_candidates if p.exists()), None)
    survival_csv = data_dir / "survival_info.csv"
    mapping_csv = data_dir / "name_mapping.csv"

    missing = []
    if meta_csv is None:
        missing.append("BraTS20 Training Metadata.csv")
    for p in [survival_csv, mapping_csv]:
        if not p.exists():
            missing.append(str(p))
    if missing:
        raise FileNotFoundError("缺少必要檔案：\n" + "\n".join(missing))

    return {
        "dataset_root": dataset_root,
        "data_dir": data_dir,
        "meta_csv": meta_csv,
        "survival_csv": survival_csv,
        "mapping_csv": mapping_csv,
    }


# =============================================================================
# 1. 資料處理
# =============================================================================
class BraTSDataContext:
    def __init__(self, paths: Dict[str, Path], crop_d: int, crop_hw: int):
        self.paths = paths
        self.crop_d = crop_d
        self.crop_hw = crop_hw
        self.df_meta = pd.read_csv(paths["meta_csv"])
        self.df_surv = pd.read_csv(paths["survival_csv"])
        self.df_surv["Survival_days"] = pd.to_numeric(
            self.df_surv["Survival_days"], errors="coerce"
        )
        self.df_map = pd.read_csv(paths["mapping_csv"])

        print(f"Metadata: {self.df_meta.shape}")
        print(f"Survival: {self.df_surv.shape}")
        print(f"Mapping : {self.df_map.shape}")
        print(
            f"Volumes: {self.df_meta['volume'].nunique()} | "
            f"Slices/Vol: {self.df_meta.groupby('volume')['slice'].count().unique()}"
        )

    def align_clinical(self, vol_id: int) -> Optional[Dict[str, float | str]]:
        try:
            patient = self.df_map.iloc[int(vol_id) - 1]["BraTS_2020_subject_ID"]
            row = self.df_surv[self.df_surv["Brats20ID"] == patient]
            if row.empty:
                return None
            age = float(row["Age"].values[0])
            surv = float(row["Survival_days"].values[0])
            if np.isnan(age) or np.isnan(surv):
                return None
            return {"age": age, "survival": surv, "patient": patient}
        except Exception:
            return None

    def valid_volume_ids(self, max_volumes: Optional[int] = None) -> List[int]:
        valid_ids: List[int] = []
        for vid in self.df_meta["volume"].unique():
            if self.align_clinical(int(vid)) is not None:
                valid_ids.append(int(vid))
        if max_volumes is not None and max_volumes > 0:
            valid_ids = valid_ids[:max_volumes]
        print(
            f"有效 Volume: {len(valid_ids)}/{self.df_meta['volume'].nunique()} "
            f"| 範例: {valid_ids[:5]}"
        )
        return valid_ids

    def get_3d_volume(self, vol_id: int, crop_d: Optional[int] = None, crop_hw: Optional[int] = None) -> np.ndarray:
        """讀取指定 Volume 的 2D h5 切片，重組為 (4, D, H, W)，並做 z-score。"""
        crop_d = self.crop_d if crop_d is None else crop_d
        crop_hw = self.crop_hw if crop_hw is None else crop_hw

        vol_df = self.df_meta[self.df_meta["volume"] == vol_id].sort_values("slice")
        stack = []
        for _, row in vol_df.iterrows():
            fp = self.paths["data_dir"] / f"volume_{int(row['volume'])}_slice_{int(row['slice'])}.h5"
            if not fp.exists():
                raise FileNotFoundError(f"找不到影像切片：{fp}")
            with h5py.File(fp, "r") as f:
                raw = f["image"][()].astype(np.float32)  # (H, W, 4)

            chw = raw.transpose(2, 0, 1)  # (4, H, W)
            for c in range(4):
                mask = chw[c] > 0
                if mask.sum() > 0:
                    mu, sg = chw[c][mask].mean(), chw[c][mask].std()
                    chw[c] = np.where(mask, (chw[c] - mu) / (sg + 1e-8), 0.0)
            stack.append(chw)

        vol = np.stack(stack, axis=1)  # (4, D, H, W)
        _, D, H, W = vol.shape

        if crop_d and D > crop_d:
            d0 = random.randint(0, D - crop_d)
            vol = vol[:, d0 : d0 + crop_d]
        if crop_hw and H > crop_hw and W > crop_hw:
            h0 = random.randint(0, H - crop_hw)
            w0 = random.randint(0, W - crop_hw)
            vol = vol[:, :, h0 : h0 + crop_hw, w0 : w0 + crop_hw]

        return np.clip(vol, -5, 5)


def survival_to_priority(days: float) -> int:
    if days < 300:
        return 2  # high priority
    if days < 600:
        return 1
    return 0


class BraTS3DDataset(Dataset):
    def __init__(self, vol_ids: List[int], ctx: BraTSDataContext):
        self.ids = vol_ids
        self.ctx = ctx

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        vid = self.ids[idx]
        img_np = self.ctx.get_3d_volume(vid)
        img = torch.from_numpy(img_np).float()  # (4, D, H, W)

        cli = self.ctx.align_clinical(vid)
        if cli is None:
            raise ValueError(f"Volume {vid} 無臨床資料")
        age = torch.tensor([cli["age"]], dtype=torch.float32)
        surv = torch.tensor([cli["survival"]], dtype=torch.float32)
        pri = torch.tensor(survival_to_priority(float(cli["survival"])), dtype=torch.long)

        # Pseudo segmentation ground truth：沿用 notebook 的強度規則。
        _, D, H, W = img.shape
        seg_gt = torch.zeros(D, H, W, dtype=torch.long)
        bright = img[1] > img[1].mean() + 1.5 * img[1].std()
        seg_gt[bright] = 3  # ET
        edema = img[2] > img[2].mean() + 0.8 * img[2].std()
        seg_gt[edema & (seg_gt == 0)] = 2  # ED
        ncr = img[0] < img[0].mean() - 0.5 * img[0].std()
        seg_gt[ncr & (seg_gt == 0)] = 1  # NCR
        return img, age, surv, pri, seg_gt


# =============================================================================
# 2. Cross-Modal Swin-Transformer 模型
# =============================================================================
class PatchEmbed3D(nn.Module):
    def __init__(self, patch_size=(2, 8, 8), in_chans=4, embed_dim=48):
        super().__init__()
        self.proj = nn.Conv3d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        B, E, Dp, Hp, Wp = x.shape
        x = x.flatten(2).transpose(1, 2)
        return self.norm(x), (Dp, Hp, Wp)


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
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if mask is not None:
            attn = attn + mask.unsqueeze(1)
        attn = self.softmax(attn)
        out = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(out)


class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)

    def forward(self, feat_t1gd, feat_t2):
        B, L, C = feat_t1gd.shape
        H = self.num_heads
        Hd = C // H
        Q = self.q_proj(self.norm_q(feat_t1gd)).reshape(B, L, H, Hd).transpose(1, 2)
        K = self.k_proj(self.norm_kv(feat_t2)).reshape(B, L, H, Hd).transpose(1, 2)
        V = self.v_proj(feat_t2).reshape(B, L, H, Hd).transpose(1, 2)
        attn = F.softmax((Q @ K.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ V).transpose(1, 2).reshape(B, L, C)
        return self.out_proj(out)


class SwinBlock3D(nn.Module):
    def __init__(self, dim, num_heads=4, window_size=(2, 7, 7), mlp_ratio=4.0, shift=False, drop=0.0):
        super().__init__()
        self.ws = window_size
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention3D(dim, window_size, num_heads)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(drop),
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
            x = torch.roll(x, shifts=(shift[0], shift[1], shift[2]), dims=(1, 2, 3))

        x = x[:, :D, :H, :W, :].contiguous().view(B, L, C)
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))
        return x


class ModalityEncoder(nn.Module):
    def __init__(self, in_chans=2, embed_dim=48, patch_size=(2, 8, 8), depth=2, num_heads=4, window_size=(2, 7, 7)):
        super().__init__()
        self.patch_embed = PatchEmbed3D(patch_size, in_chans, embed_dim)
        self.blocks = nn.ModuleList(
            [SwinBlock3D(embed_dim, num_heads, window_size=window_size, shift=(i % 2 == 1)) for i in range(depth)]
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        feat, (D, H, W) = self.patch_embed(x)
        for blk in self.blocks:
            feat = blk(feat, D, H, W)
        return self.norm(feat), (D, H, W)


class SegDecoder(nn.Module):
    def __init__(self, embed_dim=96, num_cls=4, target_d=64, target_hw=128):
        super().__init__()
        self.target_d = target_d
        self.target_hw = target_hw
        self.dec = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, num_cls),
        )

    def forward(self, fused, grid_shape):
        D, H, W = grid_shape
        B, _, _ = fused.shape
        out = self.dec(fused).transpose(1, 2).view(B, -1, D, H, W)
        return F.interpolate(
            out.float(), size=(self.target_d, self.target_hw, self.target_hw), mode="trilinear", align_corners=False
        )


class CrossModalSwinBraTS(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        E = cfg["embed_dim"]
        self.enc_t1gd = ModalityEncoder(
            in_chans=2,
            embed_dim=E,
            patch_size=cfg["patch_size"],
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
            window_size=cfg["window_size"],
        )
        self.enc_t2 = ModalityEncoder(
            in_chans=2,
            embed_dim=E,
            patch_size=cfg["patch_size"],
            depth=cfg["depth"],
            num_heads=cfg["num_heads"],
            window_size=cfg["window_size"],
        )
        self.cross_attn = CrossModalAttention(dim=E, num_heads=cfg["num_heads"])
        self.fusion_norm = nn.LayerNorm(E * 2)
        self.clinical_enc = nn.Sequential(nn.Linear(1, 32), nn.GELU(), nn.Linear(32, E), nn.LayerNorm(E))
        self.clinical_weight = nn.Parameter(torch.tensor(0.3))
        self.seg_decoder = SegDecoder(E * 2, cfg["num_seg_cls"], cfg["crop_d"], cfg["crop_hw"])
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.surv_head = nn.Sequential(
            nn.Linear(E * 3, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1), nn.Softplus()
        )
        self.pri_head = nn.Sequential(nn.Linear(E * 3, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, cfg["num_pri_cls"]))

    def forward(self, x, age):
        f_t1gd, grid = self.enc_t1gd(x[:, :2])
        f_t2, _ = self.enc_t2(x[:, 2:])
        cross = self.cross_attn(f_t1gd, f_t2)
        fused = self.fusion_norm(torch.cat([cross, f_t2], dim=-1))
        age_feat = self.clinical_enc(age)
        seg_logits = self.seg_decoder(fused, grid)
        g = self.global_pool(fused.transpose(1, 2)).squeeze(-1)
        g_with_age = torch.cat([g, age_feat * self.clinical_weight], dim=-1)
        return seg_logits, self.surv_head(g_with_age), self.pri_head(g_with_age)


# =============================================================================
# 3. Baseline 模型
# =============================================================================
class UNet3DBlock(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_c, out_c, 3, padding=1),
            nn.InstanceNorm3d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_chans=4, num_cls=4, base_c=16):
        super().__init__()
        c = base_c
        self.enc1 = UNet3DBlock(in_chans, c)
        self.enc2 = UNet3DBlock(c, c * 2)
        self.enc3 = UNet3DBlock(c * 2, c * 4)
        self.bot = UNet3DBlock(c * 4, c * 8)
        self.up3 = nn.ConvTranspose3d(c * 8, c * 4, 2, stride=2)
        self.dec3 = UNet3DBlock(c * 8, c * 4)
        self.up2 = nn.ConvTranspose3d(c * 4, c * 2, 2, stride=2)
        self.dec2 = UNet3DBlock(c * 4, c * 2)
        self.up1 = nn.ConvTranspose3d(c * 2, c, 2, stride=2)
        self.dec1 = UNet3DBlock(c * 2, c)
        self.out = nn.Conv3d(c, num_cls, 1)
        self.pool = nn.MaxPool3d(2)

    def forward(self, x, age=None):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bot(self.pool(e3))
        d3 = self.dec3(torch.cat([self.up3(b), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        seg = self.out(d1)
        B = x.shape[0]
        surv = torch.zeros(B, 1, device=x.device)
        pri = torch.zeros(B, 3, device=x.device)
        return seg, surv, pri


class ResBlock3D(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(c, c, 3, padding=1), nn.BatchNorm3d(c), nn.ReLU(inplace=True), nn.Conv3d(c, c, 3, padding=1), nn.BatchNorm3d(c)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(x + self.conv(x))


class ResNet3DRegressor(nn.Module):
    def __init__(self, in_chans=4, num_cls=4, base_c=16, crop_d=64, crop_hw=128):
        super().__init__()
        c = base_c
        self.crop_d = crop_d
        self.crop_hw = crop_hw
        self.stem = nn.Sequential(nn.Conv3d(in_chans, c, 7, stride=2, padding=3), nn.BatchNorm3d(c), nn.ReLU(inplace=True))
        self.layer1 = nn.Sequential(ResBlock3D(c), ResBlock3D(c))
        self.layer2 = nn.Sequential(nn.Conv3d(c, c * 2, 3, stride=2, padding=1), ResBlock3D(c * 2))
        self.layer3 = nn.Sequential(nn.Conv3d(c * 2, c * 4, 3, stride=2, padding=1), ResBlock3D(c * 4))
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.surv = nn.Sequential(nn.Linear(c * 4 + 1, 128), nn.ReLU(), nn.Dropout(0.3), nn.Linear(128, 1), nn.Softplus())
        self.pri = nn.Linear(c * 4 + 1, 3)
        self.seg_dummy = nn.Conv3d(c * 4, num_cls, 1)

    def forward(self, x, age):
        x = self.layer3(self.layer2(self.layer1(self.stem(x))))
        g = self.gap(x).flatten(1)
        ga = torch.cat([g, age], dim=-1)
        seg = F.interpolate(self.seg_dummy(x).float(), size=(self.crop_d, self.crop_hw, self.crop_hw), mode="trilinear", align_corners=False)
        return seg, self.surv(ga), self.pri(ga)


# =============================================================================
# 4. Loss / Train / Validate
# =============================================================================
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5, ignore_bg=True):
        super().__init__()
        self.smooth = smooth
        self.ignore_bg = ignore_bg

    def forward(self, pred, target):
        num_cls = pred.shape[1]
        pred_soft = F.softmax(pred, dim=1)
        target_oh = F.one_hot(target, num_cls).permute(0, 4, 1, 2, 3).float()
        start_cls = 1 if self.ignore_bg else 0
        dice = 0.0
        for c in range(start_cls, num_cls):
            p = pred_soft[:, c]
            g = target_oh[:, c]
            inter = (p * g).sum()
            dice += 1 - (2 * inter + self.smooth) / (p.sum() + g.sum() + self.smooth)
        return dice / (num_cls - start_cls)


dice_loss_fn = DiceLoss(ignore_bg=True)


def combined_loss(seg_logits, seg_gt, surv_pred, surv_gt, pri_pred, pri_gt, alpha=0.4, beta=0.3, gamma=0.3):
    l_dice = dice_loss_fn(seg_logits, seg_gt)
    l_ce = F.cross_entropy(seg_logits, seg_gt)
    l_seg = 0.5 * l_dice + 0.5 * l_ce
    log_pred = torch.log(surv_pred.view(-1) + 1)
    log_gt = torch.log(surv_gt.view(-1) + 1)
    l_surv = torch.sqrt(F.mse_loss(log_pred, log_gt) + 1e-6)
    l_pri = F.cross_entropy(pri_pred, pri_gt)
    total = alpha * l_seg + beta * l_surv + gamma * l_pri
    return total, l_seg.item(), l_surv.item(), l_pri.item()


def compute_dice(pred_logits, target, num_cls=4):
    pred = pred_logits.argmax(dim=1)
    dices = {}
    names = {1: "NCR", 2: "ED", 3: "ET"}
    for c, name in names.items():
        p = (pred == c).float()
        g = (target == c).float()
        inter = (p * g).sum().item()
        denom = p.sum().item() + g.sum().item()
        dices[name] = (2 * inter + 1e-5) / (denom + 1e-5)
    dices["mean"] = float(np.mean(list(dices.values())))
    return dices


def train_epoch(model, loader, optimizer, device, cfg, scaler=None):
    model.train()
    total_loss = 0.0
    for img, age, surv_gt, pri_gt, seg_gt in tqdm(loader, leave=False, desc="Train"):
        img, age = img.to(device), age.to(device)
        surv_gt, pri_gt, seg_gt = surv_gt.to(device), pri_gt.to(device), seg_gt.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.cuda.amp.autocast():
                seg, surv, pri = model(img, age)
                loss, *_ = combined_loss(seg, seg_gt, surv, surv_gt, pri, pri_gt, cfg["alpha_loss"], cfg["beta_loss"], cfg["gamma_loss"])
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            seg, surv, pri = model(img, age)
            loss, *_ = combined_loss(seg, seg_gt, surv, surv_gt, pri, pri_gt, cfg["alpha_loss"], cfg["beta_loss"], cfg["gamma_loss"])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def validate(model, loader, device, cfg):
    model.eval()
    total_loss = 0.0
    all_dice = {"NCR": [], "ED": [], "ET": [], "mean": []}
    all_surv_pred, all_surv_true = [], []
    all_pri_pred, all_pri_true = [], []

    with torch.no_grad():
        for img, age, surv_gt, pri_gt, seg_gt in tqdm(loader, leave=False, desc="Val"):
            img, age = img.to(device), age.to(device)
            surv_gt, pri_gt, seg_gt = surv_gt.to(device), pri_gt.to(device), seg_gt.to(device)
            seg, surv, pri = model(img, age)
            loss, *_ = combined_loss(seg, seg_gt, surv, surv_gt, pri, pri_gt, cfg["alpha_loss"], cfg["beta_loss"], cfg["gamma_loss"])
            total_loss += loss.item()
            d = compute_dice(seg, seg_gt)
            for k in all_dice:
                all_dice[k].append(d[k])
            all_surv_pred.append(surv.item())
            all_surv_true.append(surv_gt.item())
            all_pri_pred.append(pri.argmax(dim=1).item())
            all_pri_true.append(pri_gt.item())

    mae = float(np.mean(np.abs(np.array(all_surv_pred) - np.array(all_surv_true)))) if all_surv_true else np.nan
    acc = float(accuracy_score(all_pri_true, all_pri_pred)) if all_pri_true else np.nan
    avg_dice = {k: float(np.mean(v)) if v else np.nan for k, v in all_dice.items()}
    return {
        "val_loss": total_loss / max(1, len(loader)),
        "dice": avg_dice,
        "survival_mae": mae,
        "priority_acc": acc,
        "surv_preds": all_surv_pred,
        "surv_trues": all_surv_true,
    }


def train_model(model, train_loader, val_loader, device, cfg, output_dir, model_name="model", lr=None, weight_decay=None):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr or cfg["lr"], weight_decay=weight_decay if weight_decay is not None else cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["max_epochs"], eta_min=1e-6)
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None
    history = {
        "train_loss": [],
        "val_loss": [],
        "dice_ncr": [],
        "dice_ed": [],
        "dice_et": [],
        "dice_mean": [],
        "survival_mae": [],
        "priority_acc": [],
    }
    best_val_loss = float("inf")
    print(f"開始訓練 {model_name} | Epochs={cfg['max_epochs']}")
    for epoch in range(1, cfg["max_epochs"] + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, cfg, scaler)
        metrics = validate(model, val_loader, device, cfg)
        scheduler.step()
        history["train_loss"].append(train_loss)
        history["val_loss"].append(metrics["val_loss"])
        history["dice_ncr"].append(metrics["dice"]["NCR"])
        history["dice_ed"].append(metrics["dice"]["ED"])
        history["dice_et"].append(metrics["dice"]["ET"])
        history["dice_mean"].append(metrics["dice"]["mean"])
        history["survival_mae"].append(metrics["survival_mae"])
        history["priority_acc"].append(metrics["priority_acc"])

        flag = ""
        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]
            torch.save(model.state_dict(), output_dir / f"best_{model_name}.pth")
            flag = " ⭐ best"
        print(
            f"Epoch {epoch:02d}/{cfg['max_epochs']} | Train={train_loss:.4f} | Val={metrics['val_loss']:.4f} | "
            f"Dice={metrics['dice']['mean']:.3f} | MAE={metrics['survival_mae']:.1f}d | Acc={metrics['priority_acc']:.3f}{flag}"
        )

    torch.save(model.state_dict(), output_dir / f"last_{model_name}.pth")
    with open(output_dir / f"history_{model_name}.json", "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=2)
    return history


# =============================================================================
# 5. 視覺化
# =============================================================================
def plot_modality_preview(ctx, vol_id, cfg, output_dir):
    sample_vol = ctx.get_3d_volume(vol_id, crop_d=cfg["crop_d"], crop_hw=cfg["crop_hw"])
    mid = sample_vol.shape[1] // 2
    modality_names = ["T1", "T1Gd", "T2", "FLAIR"]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle(f"Volume {vol_id} — 中間切片四模態展示", fontsize=14, fontweight="bold")
    for i, ax in enumerate(axes):
        ax.imshow(sample_vol[i, mid], cmap="gray")
        ax.set_title(modality_names[i], fontsize=13, fontweight="bold")
        ax.axis("off")
    plt.tight_layout()
    fp = output_dir / "fig01_modality_preview.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已輸出：{fp}")


# =============================================================================
# 繪圖函數優化版 (全英文標題和邏輯修復)
# =============================================================================

def plot_training_curves(results: Dict, cfg: Dict, output_dir: Path):
    """
    輸出 fig02_training_curves.png。
    只要 results 中包含 Proposed / 3D-UNet / ResNet3D，就會在同一張圖中畫出三條線。
    """
    if len(results) <= 1:
        print("Skipping fig02_training_curves: only one model was trained. Remove --skip-baselines to include 3D-UNet and ResNet3D.")
        return

    epochs = np.arange(1, cfg["max_epochs"] + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()

    panels = [
        ("train_loss", "Training Loss", "Loss"),
        ("val_loss", "Validation Loss", "Loss"),
        ("dice_mean", "Validation Mean Dice", "Mean Dice"),
        ("survival_mae", "Survival MAE", "MAE (days)"),
    ]

    for ax, (key, title, ylabel) in zip(axes, panels):
        for model_name, history in results.items():
            values = history.get(key, [])
            if len(values) == 0:
                continue
            x = np.arange(1, len(values) + 1)
            ax.plot(x, values, marker="o", linewidth=2, label=model_name)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=9)
        ax.set_xlim(1, max(1, cfg["max_epochs"]))
        if cfg["max_epochs"] <= 15:
            ax.set_xticks(epochs)

    fig.suptitle("Proposed vs. 3D-UNet vs. ResNet3D Training Curves", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    fp = output_dir / "fig02_training_curves.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已輸出：{fp}")


def plot_dice_per_class(results: Dict, cfg: Dict, output_dir: Path):
    """
    輸出 fig03_dice_per_class.png。
    修正原本讀取不存在的 dice_classes 欄位問題，改讀 train_model 實際儲存的：
    dice_ncr、dice_ed、dice_et。
    """
    if len(results) <= 1:
        print("Skipping fig03_dice_per_class: only one model was trained. Remove --skip-baselines to include baselines.")
        return

    dice_keys = ["dice_ncr", "dice_ed", "dice_et"]
    class_names = ["NCR/NET", "ED", "ET"]
    epochs = np.arange(1, cfg["max_epochs"] + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    for ax, dice_key, class_name in zip(axes, dice_keys, class_names):
        for model_name, history in results.items():
            values = history.get(dice_key, [])
            if len(values) == 0:
                continue
            x = np.arange(1, len(values) + 1)
            ax.plot(x, values, marker="o", linewidth=2, label=model_name)
        ax.set_title(f"Dice Score: {class_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Dice")
        ax.set_ylim(0, 1.0)
        ax.grid(True, linestyle="--", alpha=0.35)
        ax.legend(fontsize=9)
        ax.set_xlim(1, max(1, cfg["max_epochs"]))
        if cfg["max_epochs"] <= 15:
            ax.set_xticks(epochs)

    fig.suptitle("Dice Performance per Tumor Sub-region", fontsize=15, fontweight="bold")
    plt.tight_layout(rect=[0, 0.03, 1, 0.92])
    fp = output_dir / "fig03_dice_per_class.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"已輸出：{fp}")


def plot_baseline_compare(results: Dict, output_dir: Path):
    """輸出最終 Mean Dice 比較長條圖。"""
    if len(results) <= 1:
        print("Skipping fig04_baseline_comparison: only one model was trained.")
        return

    model_names = list(results.keys())
    final_dice = [results[m]["dice_mean"][-1] for m in model_names]

    plt.figure(figsize=(9, 6))
    bars = plt.bar(model_names, final_dice)
    plt.title("Final Performance Comparison (Mean Dice)", fontsize=14, fontweight="bold")
    plt.ylabel("Mean Dice Score")
    plt.ylim(0, 1.0)
    plt.xticks(rotation=15, ha="right")
    plt.grid(axis="y", linestyle="--", alpha=0.3)

    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.01, f"{yval:.4f}", ha="center", va="bottom")

    plt.tight_layout()
    fp = output_dir / "fig04_baseline_comparison.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"已輸出：{fp}")


def plot_survival_scatter(model, val_loader, device, output_dir):
    model.eval()
    all_pred, all_true = [], []

    with torch.no_grad():
        for img, age, surv_gt, _, _ in val_loader:
            img = img.to(device)
            age = age.to(device)
            surv_gt = surv_gt.to(device)

            _, surv_pred, _ = model(img, age)

            all_pred.append(float(surv_pred.detach().cpu().view(-1)[0]))
            all_true.append(float(surv_gt.detach().cpu().view(-1)[0]))

    if len(all_true) < 2:
        print("驗證資料少於 2 筆，略過 survival scatter。")
        return

    fig = plt.figure(figsize=(6, 6))
    plt.scatter(all_true, all_pred, alpha=0.8)

    min_v = min(min(all_true), min(all_pred))
    max_v = max(max(all_true), max(all_pred))
    plt.plot([min_v, max_v], [min_v, max_v], linestyle="--")

    plt.xlabel("True Survival Days")
    plt.ylabel("Predicted Survival Days")
    plt.title("Survival Prediction Scatter")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    fp = output_dir / "fig05_survival_scatter.png"
    plt.savefig(fp, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"已輸出：{fp}")


# =============================================================================
# 6. 主程式
# =============================================================================
def build_argparser():
    parser = argparse.ArgumentParser(description="BraTS20 CrossModal-SwinTransformer VS Code version")
    parser.add_argument("--dataset-root", type=str, default="", help="BraTS2020 資料集根目錄；若使用 --download-kaggle 可省略")
    parser.add_argument("--download-kaggle", action="store_true", help="使用 kagglehub 下載 awsaf49/brats2020-training-data")
    parser.add_argument("--output-dir", type=str, default="outputs_brats20", help="輸出模型與圖表資料夾")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--crop-d", type=int, default=64)
    parser.add_argument("--crop-hw", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=48)
    parser.add_argument("--depth", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0, help="Windows/VS Code 建議設 0")
    parser.add_argument("--max-volumes", type=int, default=0, help="快速測試用；0 表示全部 volume")
    parser.add_argument("--skip-baselines", action="store_true", help="只訓練主模型，不訓練 Baseline")
    parser.add_argument("--skip-plots", action="store_true", help="略過圖表輸出")
    return parser


def main():
    args = build_argparser().parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    if args.download_kaggle:
        dataset_root = download_kaggle_dataset()
    else:
        if not args.dataset_root:
            raise ValueError("請提供 --dataset-root，或使用 --download-kaggle 自動下載。")
        dataset_root = Path(args.dataset_root)

    paths = resolve_dataset_paths(dataset_root)
    ctx = BraTSDataContext(paths, crop_d=args.crop_d, crop_hw=args.crop_hw)

    cfg = {
        "embed_dim": args.embed_dim,
        "patch_size": (2, 8, 8),
        "depth": args.depth,
        "num_heads": args.num_heads,
        "window_size": (2, 7, 7),
        "num_seg_cls": 4,
        "num_pri_cls": 3,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "max_epochs": args.epochs,
        "batch_size": args.batch_size,
        "crop_d": args.crop_d,
        "crop_hw": args.crop_hw,
        "seed": args.seed,
        "alpha_loss": 0.4,
        "beta_loss": 0.3,
        "gamma_loss": 0.3,
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"裝置: {device} | CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"顯存: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    valid_ids = ctx.valid_volume_ids(max_volumes=args.max_volumes if args.max_volumes > 0 else None)
    if len(valid_ids) < 2:
        raise ValueError("有效資料少於 2 筆，無法切分 train/val。")

    random.shuffle(valid_ids)
    split = max(1, int(len(valid_ids) * 0.8))
    train_ids, val_ids = valid_ids[:split], valid_ids[split:]
    if len(val_ids) == 0:
        val_ids = train_ids[-1:]
        train_ids = train_ids[:-1]

    train_ds = BraTS3DDataset(train_ids, ctx)
    val_ds = BraTS3DDataset(val_ids, ctx)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=args.num_workers, pin_memory=torch.cuda.is_available())
    print(f"訓練集: {len(train_ds)} | 驗證集: {len(val_ds)}")

    if not args.skip_plots:
        plot_modality_preview(ctx, valid_ids[0], cfg, output_dir)

    model = CrossModalSwinBraTS(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"CrossModal-SwinTransformer 可訓練參數: {n_params:,} ({n_params / 1e6:.2f}M)")

    # Forward test
    with torch.no_grad():
        dummy_img = torch.randn(1, 4, cfg["crop_d"], cfg["crop_hw"], cfg["crop_hw"], device=device)
        dummy_age = torch.tensor([[55.0]], device=device)
        seg, surv, pri = model(dummy_img, dummy_age)
        print(f"Forward test | seg={tuple(seg.shape)} | surv={tuple(surv.shape)} | pri={tuple(pri.shape)}")

    # 執行主模型訓練
    results = {}
    results["Swin-Transformer (Proposed)"] = train_model(
        model, train_loader, val_loader, device, cfg, output_dir, model_name="swin_final"
    )

    # 只有在未跳過 Baseline 且模型數據正常時才執行
    if not args.skip_baselines:
        print("\n[Comparison Mode] Training Baseline Models...")
        baselines = {
            "3D-UNet": UNet3D(base_c=16).to(device),
            "ResNet3D": ResNet3DRegressor(base_c=16, crop_d=cfg["crop_d"], crop_hw=cfg["crop_hw"]).to(device),
        }
        for name, bmodel in baselines.items():
            try:
                results[name] = train_model(
                    bmodel, train_loader, val_loader, device, cfg, output_dir, 
                    model_name=name.lower().replace("-", "_"), lr=1e-3, weight_decay=0.0
                )
            except Exception as e:
                print(f"Warning: Baseline {name} failed: {e}")

    # 存檔及繪圖
    with open(output_dir / "all_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    if not args.skip_plots:
        # 這裡會根據 results 數量自動判斷要不要畫對比圖
        plot_training_curves(results, cfg, output_dir)
        plot_dice_per_class(results, cfg, output_dir)
        plot_baseline_compare(results, output_dir)
        plot_survival_scatter(model, val_loader, device, output_dir)

    print("\n" + "="*70)
    print("Execution Summary")
    for name, h in results.items():
        print(f"\n{name}")
        print(f"  Mean Dice    : {h['dice_mean'][-1]:.4f}")
        print(f"  NCR Dice     : {h['dice_ncr'][-1]:.4f}")
        print(f"  ED Dice      : {h['dice_ed'][-1]:.4f}")
        print(f"  ET Dice      : {h['dice_et'][-1]:.4f}")
        print(f"  Survival MAE : {h['survival_mae'][-1]:.1f} days")
        print(f"  Priority Acc : {h['priority_acc'][-1]:.4f}")
        print(f"  Val Loss     : {h['val_loss'][-1]:.4f}")
    print(f"\n輸出位置：{output_dir.resolve()}")
    print("=" * 70)


if __name__ == "__main__":
    main()
