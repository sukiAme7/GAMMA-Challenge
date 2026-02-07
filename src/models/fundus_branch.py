"""
GAMMA Challenge - Fundus (2D) 分支
使用EfficientNet-B4/ConvNeXt/Swin作为backbone
支持从本地加载 HuggingFace 模型
"""
import os
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
        
        # NOTE: 支持从本地目录加载 HuggingFace 模型
        # 如果配置了本地模型路径且目录存在，则优先从本地加载
        local_model_dir = getattr(cfg, 'FUNDUS_LOCAL_MODEL_DIR', None)
        
        if pretrained and local_model_dir and Path(local_model_dir).exists():
            # 从本地 HuggingFace 目录加载（推荐方式）
            print(f"[INFO] 从本地目录加载预训练模型: {local_model_dir}")
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=False,
                num_classes=0,
                global_pool="avg"
            )
            # 加载本地权重文件
            weights_file = self._find_weights_file(local_model_dir)
            if weights_file:
                state_dict = self._load_weights(weights_file)
                # 过滤掉分类头的权重
                state_dict = {k: v for k, v in state_dict.items() 
                             if not k.startswith('classifier') and not k.startswith('fc')}
                self.backbone.load_state_dict(state_dict, strict=False)
                print(f"[INFO] 成功加载本地预训练权重: {weights_file}")
            else:
                print(f"[WARN] 本地目录中未找到权重文件，使用随机初始化")
        else:
            # 尝试从网络下载，失败则使用随机初始化
            try:
                self.backbone = timm.create_model(
                    backbone_name,
                    pretrained=pretrained,
                    num_classes=0,
                    global_pool="avg"
                )
                if pretrained:
                    print(f"[INFO] 从网络加载预训练权重成功")
            except Exception as e:
                print(f"[WARN] 无法下载预训练权重 ({type(e).__name__}: {e})")
                print(f"[INFO] 使用随机初始化权重")
                self.backbone = timm.create_model(
                    backbone_name,
                    pretrained=False,
                    num_classes=0,
                    global_pool="avg"
                )
        
        # 获取特征维度
        self.feature_dim = self.backbone.num_features
        
        # 冻结部分层
        if freeze_layers > 0:
            self._freeze_layers(freeze_layers)
        
        print(f"[INFO] FundusBranch: {backbone_name}, feature_dim={self.feature_dim}")
    
    def _find_weights_file(self, model_dir: str) -> str:
        """在模型目录中查找权重文件"""
        model_dir = Path(model_dir)
        
        # 按优先级查找不同格式的权重文件
        candidates = [
            "model.safetensors",
            "pytorch_model.bin", 
            "model.bin",
        ]
        
        for filename in candidates:
            filepath = model_dir / filename
            if filepath.exists():
                return str(filepath)
        
        # 查找 snapshots 子目录（HuggingFace 缓存格式）
        snapshots_dir = model_dir / "snapshots"
        if snapshots_dir.exists():
            for snapshot in snapshots_dir.iterdir():
                if snapshot.is_dir():
                    for filename in candidates:
                        filepath = snapshot / filename
                        if filepath.exists():
                            return str(filepath)
        
        return None
    
    def _load_weights(self, weights_file: str) -> dict:
        """加载权重文件，支持 safetensors 和 pytorch 格式"""
        if weights_file.endswith('.safetensors'):
            try:
                from safetensors.torch import load_file
                return load_file(weights_file)
            except ImportError:
                print("[WARN] 未安装 safetensors，请运行: pip install safetensors")
                return {}
        else:
            return torch.load(weights_file, map_location="cpu", weights_only=True)
    
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
