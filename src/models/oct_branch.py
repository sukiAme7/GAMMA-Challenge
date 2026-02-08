"""
GAMMA Challenge - OCT (3D/2.5D) 分支
支持2.5D切片融合和3D卷积两种模式
"""
import torch
import torch.nn as nn
import timm
import os
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
        
        # 使用timm创建backbone
        # 优先尝试从本地加载
        local_model_dir = getattr(cfg, 'FUNDUS_LOCAL_MODEL_DIR', None)
        
        if pretrained and local_model_dir and Path(local_model_dir).exists():
            print(f"[INFO] OCTBranch2D: 从本地目录加载预训练模型: {local_model_dir}")
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=False,
                in_chans=1,
                num_classes=0,
                global_pool="avg"
            )
            
            # 加载本地权重并适配单通道
            weights_file = self._find_weights_file(local_model_dir)
            if weights_file:
                state_dict = self._load_weights(weights_file)
                # 过滤分类头
                state_dict = {k: v for k, v in state_dict.items() 
                             if not k.startswith('classifier') and not k.startswith('fc')}
                
                # 适配单通道 (3->1)
                self._adapt_input_conv(state_dict)
                
                self.backbone.load_state_dict(state_dict, strict=False)
                print(f"[INFO] OCTBranch2D: 成功加载本地权重并适配单通道: {weights_file}")
            else:
                print(f"[WARN] OCTBranch2D: 本地未找到权重文件，使用随机初始化")
        else:
            # 尝试从网络加载
            try:
                self.backbone = timm.create_model(
                    backbone_name,
                    pretrained=pretrained,
                    in_chans=1,
                    num_classes=0,
                    global_pool="avg"
                )
            except Exception as e:
                print(f"[WARN] OCTBranch2D: 无法下载预训练权重 ({e})，使用随机初始化")
                self.backbone = timm.create_model(
                    backbone_name,
                    pretrained=False,
                    in_chans=1,
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
    
    def _find_weights_file(self, model_dir: str) -> str:
        """在模型目录中查找权重文件"""
        model_dir = Path(model_dir)
        candidates = ["model.safetensors", "pytorch_model.bin", "model.bin"]
        
        for filename in candidates:
            filepath = model_dir / filename
            if filepath.exists():
                return str(filepath)
        
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
        """加载权重文件"""
        if weights_file.endswith('.safetensors'):
            try:
                from safetensors.torch import load_file
                return load_file(weights_file)
            except ImportError:
                print("[WARN] 未安装 safetensors")
                return {}
        else:
            return torch.load(weights_file, map_location="cpu", weights_only=True)

    def _adapt_input_conv(self, state_dict: dict):
        """适配输入层卷积通道数 (3->1)"""
        # EfficientNet series
        conv_names = ['conv_stem.weight', 'stem.0.weight']
        
        for name in conv_names:
            if name in state_dict:
                w = state_dict[name]
                if w.shape[1] == 3:  # (Out, 3, K, K)
                    print(f"[INFO] Converting {name} from 3 channels to 1 channel (summing)")
                    state_dict[name] = w.sum(dim=1, keepdim=True)
                break

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


def get_oct_branch(mode: str = None, pretrained: bool = True) -> nn.Module:
    """工厂函数：根据配置返回OCT分支"""
    mode = mode or cfg.OCT_MODE
    
    if mode == "2.5d":
        return OCTBranch2D(pretrained=pretrained)
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
