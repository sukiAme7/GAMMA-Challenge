"""
GAMMA Challenge - 多模态融合模块
支持简单拼接和Cross-Attention融合
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import cfg


class ConcatFusion(nn.Module):
    """
    简单拼接融合
    将两个模态的特征直接拼接后通过MLP
    """
    
    def __init__(
        self,
        fundus_dim: int,
        oct_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.mlp = nn.Sequential(
            nn.Linear(fundus_dim + oct_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        
        self.output_dim = hidden_dim
    
    def forward(
        self,
        fundus_feat: torch.Tensor,
        oct_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            fundus_feat: (B, fundus_dim)
            oct_feat: (B, oct_dim)
        
        Returns:
            fused: (B, hidden_dim)
        """
        concat = torch.cat([fundus_feat, oct_feat], dim=1)
        return self.mlp(concat)


class CrossAttentionFusion(nn.Module):
    """
    Cross-Attention融合模块
    让两个模态互相attend，学习更丰富的交互特征
    """
    
    def __init__(
        self,
        fundus_dim: int,
        oct_dim: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        dropout: float = 0.5,
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # 投影到相同维度
        self.fundus_proj = nn.Linear(fundus_dim, hidden_dim)
        self.oct_proj = nn.Linear(oct_dim, hidden_dim)
        
        # Cross-Attention: Fundus attends to OCT
        self.cross_attn_f2o = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-Attention: OCT attends to Fundus
        self.cross_attn_o2f = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # LayerNorms
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        
        # 最终融合MLP
        self.fusion_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.output_dim = hidden_dim
    
    def forward(
        self,
        fundus_feat: torch.Tensor,
        oct_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            fundus_feat: (B, fundus_dim)
            oct_feat: (B, oct_dim)
        
        Returns:
            fused: (B, hidden_dim)
        """
        # 投影到相同维度
        f = self.fundus_proj(fundus_feat)  # (B, hidden_dim)
        o = self.oct_proj(oct_feat)        # (B, hidden_dim)
        
        # 添加token维度 for attention: (B, 1, hidden_dim)
        f = f.unsqueeze(1)
        o = o.unsqueeze(1)
        
        # Cross-Attention
        # Fundus attends to OCT (query=f, key=value=o)
        f_attended, _ = self.cross_attn_f2o(f, o, o)
        f_attended = self.ln1(f + f_attended)
        
        # OCT attends to Fundus (query=o, key=value=f)
        o_attended, _ = self.cross_attn_o2f(o, f, f)
        o_attended = self.ln2(o + o_attended)
        
        # 拼接并融合
        f_attended = f_attended.squeeze(1)  # (B, hidden_dim)
        o_attended = o_attended.squeeze(1)
        
        fused = torch.cat([f_attended, o_attended], dim=1)  # (B, 2*hidden_dim)
        fused = self.fusion_mlp(fused)  # (B, hidden_dim)
        
        return fused


def get_fusion_module(
    fundus_dim: int,
    oct_dim: int,
    fusion_type: str = None,
) -> nn.Module:
    """工厂函数：根据配置返回融合模块"""
    fusion_type = fusion_type or cfg.FUSION_TYPE
    hidden_dim = cfg.FUSION_HIDDEN_DIM
    dropout = cfg.FUSION_DROPOUT
    
    if fusion_type == "concat":
        return ConcatFusion(fundus_dim, oct_dim, hidden_dim, dropout)
    elif fusion_type == "attention":
        return CrossAttentionFusion(fundus_dim, oct_dim, hidden_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown fusion type: {fusion_type}")


if __name__ == "__main__":
    # 测试
    fundus_feat = torch.randn(4, 1792)  # EfficientNet-B4特征
    oct_feat = torch.randn(4, 1792)     # OCT特征
    
    # 测试Concat融合
    concat_fusion = ConcatFusion(1792, 1792)
    out = concat_fusion(fundus_feat, oct_feat)
    print(f"Concat Fusion: {fundus_feat.shape} + {oct_feat.shape} -> {out.shape}")
    
    # 测试Cross-Attention融合
    attn_fusion = CrossAttentionFusion(1792, 1792)
    out = attn_fusion(fundus_feat, oct_feat)
    print(f"Cross-Attention Fusion: {fundus_feat.shape} + {oct_feat.shape} -> {out.shape}")
