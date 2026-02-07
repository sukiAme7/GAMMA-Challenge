"""
GAMMA Challenge 配置文件
包含所有训练和模型相关的超参数
"""
import os
from pathlib import Path


class Config:
    """训练配置类"""
    
    # ========================= 数据路径 =========================
    DATA_ROOT = Path(r"d:\code\Gamma_Challenge\Glaucoma_grading\Glaucoma_grading")
    TRAIN_DIR = DATA_ROOT / "Training"
    VAL_DIR = DATA_ROOT / "Validation"
    TEST_DIR = DATA_ROOT / "Testing"
    
    TRAIN_LABEL_FILE = TRAIN_DIR / "glaucoma_grading_training_GT.xlsx"
    VAL_LABEL_FILE = VAL_DIR / "glaucoma_grading.xlsx"
    TEST_LABEL_FILE = TEST_DIR / "glaucoma_grading_testing_GT.xlsx"
    
    # ========================= 模型配置 =========================
    NUM_CLASSES = 3  # 0: Non, 1: Early, 2: Mid/Advanced
    
    # Fundus分支 (2D)
    FUNDUS_BACKBONE = "efficientnet_b4"  # 可选: efficientnet_b4, convnext_base, swin_base
    FUNDUS_PRETRAINED = True
    FUNDUS_IMG_SIZE = 512
    
    # OCT分支 (3D/2.5D)
    OCT_MODE = "2.5d"  # 可选: "3d", "2.5d"
    OCT_NUM_SLICES = 64  # 选取的切片数量
    OCT_IMG_SIZE = 256
    OCT_BACKBONE = "efficientnet_b4"  # 2.5D模式下使用的backbone
    
    # 融合模块
    FUSION_TYPE = "attention"  # 可选: "concat", "attention"
    FUSION_HIDDEN_DIM = 512
    FUSION_DROPOUT = 0.5
    
    # ========================= 训练配置 =========================
    EPOCHS = 100
    BATCH_SIZE = 4
    ACCUMULATION_STEPS = 2  # 梯度累积，等效 batch_size = 8
    
    # 优化器
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.05
    
    # 学习率调度
    SCHEDULER = "cosine"  # 可选: "cosine", "step"
    WARMUP_EPOCHS = 5
    MIN_LR = 1e-6
    
    # 损失函数
    LOSS_TYPE = "focal"  # 可选: "ce", "focal"
    FOCAL_GAMMA = 2.0
    LABEL_SMOOTHING = 0.1
    
    # 类别权重 (处理不平衡)
    CLASS_WEIGHTS = [1.0, 2.0, 2.0]  # 根据实际分布调整
    
    # ========================= 数据增强 =========================
    AUGMENTATION = True
    MIXUP_ALPHA = 0.2
    CUTMIX_ALPHA = 1.0
    MIXUP_PROB = 0.5
    
    # ========================= 其他 =========================
    SEED = 42
    NUM_WORKERS = 4
    DEVICE = "cuda"
    
    # 保存路径
    OUTPUT_DIR = Path(r"d:\code\Gamma_Challenge\outputs")
    CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
    LOG_DIR = OUTPUT_DIR / "logs"
    
    # 早停
    EARLY_STOPPING_PATIENCE = 15
    
    # K折交叉验证
    USE_KFOLD = False
    NUM_FOLDS = 5
    
    @classmethod
    def setup_dirs(cls):
        """创建必要的目录"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        cls.CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        cls.LOG_DIR.mkdir(parents=True, exist_ok=True)


# 全局配置实例
cfg = Config()
