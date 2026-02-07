"""
GAMMA Challenge - OCT (3D/2.5D) 分支
支持2.5D切片融合和3D卷积两种模式
"""
import torch
import torch.nn as nn
import timm
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import cfg


class OCTBranch2D(nn.Module):
    """
    2.5D OCT处理分支
    将OCT切片序列通过2D backbone提取特征后融合
    """
    
    def __init__(
        self,
        backbone_name: str = None,
        pretrained: bool = True,
        num_slices: int = None,
        aggregation: str = "attention",
    ):
        """
        Args:
            backbone_name: timm模型名称
            pretrained: 是否使用预训练权重
            num_slices: OCT切片数量
            aggregation: 切片特征聚合方式 ("mean", "max", "attention")
        """
        super().__init__()
        
        backbone_name = backbone_name or cfg.OCT_BACKBONE
        num_slices = num_slices or cfg.OCT_NUM_SLICES
        self.num_slices = num_slices
        self.aggregation = aggregation
        
        # 使用timm创建backbone (修改输入通道为1)
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            in_chans=1,  # 灰度图
            num_classes=0,
            global_pool="avg"
        )
        
        self.feature_dim = self.backbone.num_features
        
        # 注意力聚合
        if aggregation == "attention":
            self.attention = nn.Sequential(
                nn.Linear(self.feature_dim, self.feature_dim // 4),
                nn.Tanh(),
                nn.Linear(self.feature_dim // 4, 1),
            )
        
        print(f"[INFO] OCTBranch2D: {backbone_name}, slices={num_slices}, agg={aggregation}, feature_dim={self.feature_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: OCT切片序列 (B, num_slices, 1, H, W)
        
        Returns:
            features: (B, feature_dim)
        """
        B, N, C, H, W = x.shape
        
        # 合并batch和slice维度: (B*N, 1, H, W)
        x = x.view(B * N, C, H, W)
        
        # 提取每个切片的特征: (B*N, feature_dim)
        features = self.backbone(x)
        
        # 恢复维度: (B, N, feature_dim)
        features = features.view(B, N, -1)
        
        # 聚合切片特征
        if self.aggregation == "mean":
            out = features.mean(dim=1)  # (B, feature_dim)
        elif self.aggregation == "max":
            out = features.max(dim=1)[0]
        elif self.aggregation == "attention":
            # 计算注意力权重
            attn_weights = self.attention(features)  # (B, N, 1)
            attn_weights = torch.softmax(attn_weights, dim=1)
            out = (features * attn_weights).sum(dim=1)  # (B, feature_dim)
        else:
            raise ValueError(f"Unknown aggregation: {self.aggregation}")
        
        return out


class OCTBranch3D(nn.Module):
    """
    3D OCT处理分支
    使用3D卷积直接处理OCT体积
    """
    
    def __init__(
        self,
        feature_dim: int = 512,
    ):
        super().__init__()
        
        # 简化的3D ResNet
        self.conv3d = nn.Sequential(
            # Block 1: (1, 64, 256, 256) -> (32, 32, 128, 128)
            nn.Conv3d(1, 32, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            
            # Block 2: -> (64, 16, 64, 64)
            nn.Conv3d(32, 64, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            
            # Block 3: -> (128, 8, 32, 32)
            nn.Conv3d(64, 128, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            
            # Block 4: -> (256, 4, 16, 16)
            nn.Conv3d(128, 256, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            
            # Block 5: -> (512, 2, 8, 8)
            nn.Conv3d(256, 512, kernel_size=3, stride=(2, 2, 2), padding=1),
            nn.BatchNorm3d(512),
            nn.ReLU(inplace=True),
        )
        
        self.pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512, feature_dim)
        self.feature_dim = feature_dim
        
        print(f"[INFO] OCTBranch3D: feature_dim={feature_dim}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: OCT体积 (B, num_slices, 1, H, W) 或 (B, 1, D, H, W)
        
        Returns:
            features: (B, feature_dim)
        """
        # 调整维度: (B, N, 1, H, W) -> (B, 1, N, H, W)
        if x.dim() == 5 and x.shape[2] == 1:
            x = x.squeeze(2).unsqueeze(1)  # (B, 1, N, H, W)
        
        x = self.conv3d(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def get_oct_branch(mode: str = None) -> nn.Module:
    """工厂函数：根据配置返回OCT分支"""
    mode = mode or cfg.OCT_MODE
    
    if mode == "2.5d":
        return OCTBranch2D()
    elif mode == "3d":
        return OCTBranch3D()
    else:
        raise ValueError(f"Unknown OCT mode: {mode}")


if __name__ == "__main__":
    # 测试2.5D
    model_2d = OCTBranch2D("efficientnet_b4", pretrained=False, num_slices=64)
    x = torch.randn(2, 64, 1, 256, 256)
    out = model_2d(x)
    print(f"2.5D Input: {x.shape} -> Output: {out.shape}")
    
    # 测试3D
    model_3d = OCTBranch3D()
    out = model_3d(x)
    print(f"3D Input: {x.shape} -> Output: {out.shape}")
