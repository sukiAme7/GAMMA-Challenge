"""
GAMMA Challenge - 完整模型
集成Fundus分支、OCT分支、融合模块和分类头
"""
import torch
import torch.nn as nn
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import cfg
from models.fundus_branch import FundusBranch
from models.oct_branch import get_oct_branch
from models.fusion import get_fusion_module


class GammaNet(nn.Module):
    """
    GAMMA挑战完整模型
    多模态青光眼分级网络
    """
    
    def __init__(
        self,
        num_classes: int = None,
        fundus_backbone: str = None,
        oct_mode: str = None,
        fusion_type: str = None,
        pretrained: bool = True,
    ):
        super().__init__()
        
        num_classes = num_classes or cfg.NUM_CLASSES
        
        # Fundus分支
        self.fundus_branch = FundusBranch(
            backbone_name=fundus_backbone,
            pretrained=pretrained,
        )
        
        # OCT分支
        self.oct_branch = get_oct_branch(mode=oct_mode)
        
        # 融合模块
        self.fusion = get_fusion_module(
            fundus_dim=self.fundus_branch.feature_dim,
            oct_dim=self.oct_branch.feature_dim,
            fusion_type=fusion_type,
        )
        
        # 分类头
        self.classifier = nn.Sequential(
            nn.Linear(self.fusion.output_dim, num_classes),
        )
        
        print(f"[INFO] GammaNet initialized:")
        print(f"  - Fundus: {self.fundus_branch.feature_dim}d")
        print(f"  - OCT: {self.oct_branch.feature_dim}d")
        print(f"  - Fusion: {self.fusion.output_dim}d -> {num_classes} classes")
    
    def forward(
        self,
        fundus: torch.Tensor,
        oct: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            fundus: Fundus图像 (B, 3, H, W)
            oct: OCT切片序列 (B, num_slices, 1, H, W)
        
        Returns:
            logits: (B, num_classes)
        """
        # 提取各模态特征
        fundus_feat = self.fundus_branch(fundus)  # (B, fundus_dim)
        oct_feat = self.oct_branch(oct)            # (B, oct_dim)
        
        # 融合
        fused = self.fusion(fundus_feat, oct_feat)  # (B, fusion_dim)
        
        # 分类
        logits = self.classifier(fused)  # (B, num_classes)
        
        return logits
    
    def get_features(
        self,
        fundus: torch.Tensor,
        oct: torch.Tensor,
    ) -> dict:
        """获取中间特征，用于分析和可视化"""
        fundus_feat = self.fundus_branch(fundus)
        oct_feat = self.oct_branch(oct)
        fused = self.fusion(fundus_feat, oct_feat)
        
        return {
            "fundus": fundus_feat,
            "oct": oct_feat,
            "fused": fused,
        }


def build_model(pretrained: bool = True) -> GammaNet:
    """构建模型的工厂函数"""
    return GammaNet(
        num_classes=cfg.NUM_CLASSES,
        fundus_backbone=cfg.FUNDUS_BACKBONE,
        oct_mode=cfg.OCT_MODE,
        fusion_type=cfg.FUSION_TYPE,
        pretrained=pretrained,
    )


if __name__ == "__main__":
    # 测试完整模型
    model = build_model(pretrained=False)
    
    fundus = torch.randn(2, 3, 512, 512)
    oct = torch.randn(2, 64, 1, 256, 256)
    
    logits = model(fundus, oct)
    print(f"Fundus: {fundus.shape}")
    print(f"OCT: {oct.shape}")
    print(f"Logits: {logits.shape}")
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total params: {total_params / 1e6:.2f}M")
    print(f"Trainable params: {trainable_params / 1e6:.2f}M")
