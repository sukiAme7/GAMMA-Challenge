"""
GAMMA Challenge 数据加载模块
支持多模态数据加载：2D Fundus + 3D OCT
"""
import os
from pathlib import Path
from typing import Tuple, List, Optional, Dict

import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import cfg


class GammaDataset(Dataset):
    """
    GAMMA挑战数据集
    加载配对的Fundus图像和OCT切片序列
    """
    
    def __init__(
        self,
        data_dir: Path,
        label_file: Path,
        mode: str = "train",
        fundus_transform: Optional[A.Compose] = None,
        oct_transform: Optional[A.Compose] = None,
    ):
        """
        Args:
            data_dir: 数据目录 (包含 multi-modality_images 子目录)
            label_file: 标签文件路径 (.xlsx)
            mode: 模式 ("train", "val", "test")
            fundus_transform: Fundus图像增强
            oct_transform: OCT切片增强
        """
        self.data_dir = Path(data_dir)
        self.image_dir = self.data_dir / "multi-modality_images"
        self.mode = mode
        self.fundus_transform = fundus_transform
        self.oct_transform = oct_transform
        
        # 加载标签
        self.df = pd.read_excel(label_file)
        # NOTE: 假设第一列是样本ID，第二列是标签
        self.df.columns = ["sample_id", "label"] if len(self.df.columns) == 2 else self.df.columns
        
        # 获取样本ID列表
        self.sample_ids = self.df.iloc[:, 0].astype(str).str.zfill(4).tolist()
        self.labels = self.df.iloc[:, 1].tolist()
        
        # 验证数据完整性
        self._validate_data()
    
    def _validate_data(self):
        """验证数据文件是否存在"""
        valid_samples = []
        valid_labels = []
        
        for sid, label in zip(self.sample_ids, self.labels):
            sample_dir = self.image_dir / sid
            fundus_path = sample_dir / f"{sid}.jpg"
            oct_dir = sample_dir / sid
            
            if fundus_path.exists() and oct_dir.exists():
                valid_samples.append(sid)
                valid_labels.append(label)
            else:
                print(f"[WARN] 样本 {sid} 数据不完整，已跳过")
        
        self.sample_ids = valid_samples
        self.labels = valid_labels
        print(f"[INFO] 加载 {len(self.sample_ids)} 个有效样本 ({self.mode})")
    
    def __len__(self) -> int:
        return len(self.sample_ids)
    
    def _load_fundus(self, sample_id: str) -> np.ndarray:
        """加载Fundus图像"""
        fundus_path = self.image_dir / sample_id / f"{sample_id}.jpg"
        img = cv2.imread(str(fundus_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # 调整大小
        img = cv2.resize(img, (cfg.FUNDUS_IMG_SIZE, cfg.FUNDUS_IMG_SIZE))
        return img
    
    def _load_oct_slices(self, sample_id: str) -> np.ndarray:
        """
        加载OCT切片序列
        
        Returns:
            OCT volume: (num_slices, H, W) for 2.5D or (D, H, W) for 3D
        """
        oct_dir = self.image_dir / sample_id / sample_id
        
        # 获取所有切片文件并排序
        slice_files = sorted(
            oct_dir.glob("*_image.jpg"),
            key=lambda x: int(x.stem.split("_")[0])
        )
        
        total_slices = len(slice_files)
        
        if cfg.OCT_MODE == "2.5d":
            # 2.5D模式：均匀采样指定数量的切片
            indices = np.linspace(0, total_slices - 1, cfg.OCT_NUM_SLICES, dtype=int)
            selected_files = [slice_files[i] for i in indices]
        else:
            # 3D模式：加载所有切片并调整
            selected_files = slice_files
        
        slices = []
        for f in selected_files:
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (cfg.OCT_IMG_SIZE, cfg.OCT_IMG_SIZE))
            slices.append(img)
        
        # Stack成 (num_slices, H, W)
        volume = np.stack(slices, axis=0)
        return volume
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample_id = self.sample_ids[idx]
        label = self.labels[idx]
        
        # 加载Fundus
        fundus = self._load_fundus(sample_id)
        
        # 加载OCT
        oct_volume = self._load_oct_slices(sample_id)
        
        # 应用数据增强
        if self.fundus_transform:
            fundus = self.fundus_transform(image=fundus)["image"]
        else:
            # 默认归一化
            fundus = torch.from_numpy(fundus.transpose(2, 0, 1).astype(np.float32) / 255.0)
        
        if self.oct_transform:
            # 对每个切片应用相同的增强
            aug_slices = []
            for i in range(oct_volume.shape[0]):
                aug = self.oct_transform(image=oct_volume[i])["image"]
                aug_slices.append(aug)
            oct_volume = torch.stack(aug_slices, dim=0)
        else:
            # 默认归一化: (num_slices, H, W) -> (num_slices, 1, H, W) for 2.5D
            oct_volume = torch.from_numpy(oct_volume.astype(np.float32) / 255.0)
            if cfg.OCT_MODE == "2.5d":
                oct_volume = oct_volume.unsqueeze(1)  # (N, 1, H, W)
        
        return {
            "fundus": fundus,
            "oct": oct_volume,
            "label": torch.tensor(label, dtype=torch.long),
            "sample_id": sample_id,
        }


def get_fundus_transforms(mode: str = "train") -> A.Compose:
    """获取Fundus图像增强"""
    if mode == "train" and cfg.AUGMENTATION:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)},
                scale=(0.85, 1.15),
                rotate=(-30, 30),
                mode=cv2.BORDER_CONSTANT,
                p=0.5
            ),
            A.OneOf([
                A.GaussNoise(std_range=(0.02, 0.1)),  # 新版本 API
                A.GaussianBlur(blur_limit=(3, 5)),
            ], p=0.3),
            A.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1,
                p=0.5
            ),
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            ToTensorV2(),
        ])


def get_oct_transforms(mode: str = "train") -> A.Compose:
    """获取OCT切片增强"""
    if mode == "train" and cfg.AUGMENTATION:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.Affine(
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
                scale=(0.9, 1.1),
                rotate=(-10, 10),
                mode=cv2.BORDER_CONSTANT,
                p=0.3
            ),
            A.GaussNoise(std_range=(0.01, 0.05), p=0.2),  # 新版本 API
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])


def get_dataloader(
    data_dir: Path,
    label_file: Path,
    mode: str = "train",
    batch_size: int = None,
    num_workers: int = None,
) -> DataLoader:
    """创建DataLoader"""
    batch_size = batch_size or cfg.BATCH_SIZE
    num_workers = num_workers or cfg.NUM_WORKERS
    
    dataset = GammaDataset(
        data_dir=data_dir,
        label_file=label_file,
        mode=mode,
        fundus_transform=get_fundus_transforms(mode),
        oct_transform=get_oct_transforms(mode),
    )
    
    shuffle = (mode == "train")
    drop_last = (mode == "train")
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
    )


if __name__ == "__main__":
    # 测试数据加载
    train_loader = get_dataloader(
        data_dir=cfg.TRAIN_DIR,
        label_file=cfg.TRAIN_LABEL_FILE,
        mode="train",
        batch_size=2,
    )
    
    for batch in train_loader:
        print(f"Fundus shape: {batch['fundus'].shape}")  # (B, 3, 512, 512)
        print(f"OCT shape: {batch['oct'].shape}")        # (B, 64, 1, 256, 256)
        print(f"Labels: {batch['label']}")
        break
