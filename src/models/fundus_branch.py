"""
GAMMA Challenge - Fundus (2D) 分支
使用EfficientNet-B4/ConvNeXt/Swin作为backbone
"""
import torch
import torch.nn as nn
import timm

from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import cfg


class FundusBranch(nn.Module):
    """
    2D Fundus图像处理分支
    使用预训练的backbone提取特征
    """
    
    def __init__(
        self,
        backbone_name: str = None,
        pretrained: bool = True,
        freeze_layers: int = 0,
    ):
        """
        Args:
            backbone_name: timm模型名称
            pretrained: 是否使用预训练权重
            freeze_layers: 冻结前N层 (0表示不冻结)
        """
        super().__init__()
        
        backbone_name = backbone_name or cfg.FUNDUS_BACKBONE
        
        # 使用timm创建backbone
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            num_classes=0,  # 移除分类头，只保留特征提取
            global_pool="avg"
        )
        
        # 获取特征维度
        self.feature_dim = self.backbone.num_features
        
        # 冻结部分层
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        
        print(f"[INFO] FundusBranch: {backbone_name}, feature_dim={self.feature_dim}")
    
    def _freeze_layers(self, num_layers: int):
        """冻结backbone的前N层"""
        children = list(self.backbone.children())
        for i, child in enumerate(children[:num_layers]):
            for param in child.parameters():
                param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Fundus图像 (B, 3, H, W)
        
        Returns:
            features: (B, feature_dim)
        """
        return self.backbone(x)


if __name__ == "__main__":
    
    model = FundusBranch("efficientnet_b4", pretrained=False)
    x = torch.randn(2, 3, 512, 512)
    out = model(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    # 预期: Input: (2, 3, 512, 512) -> Output: (2, 1792)
