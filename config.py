# 项目配置文件
import os
import torch

# 项目根目录
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# 设备检测与配置
def get_device():
    # 检测可用设备，优先 XPU，其次 CUDA，最后 CPU
    # 尝试导入 ipex 注册 XPU 设备
    try:
        import intel_extension_for_pytorch as ipex  # noqa: F401
    except ImportError:
        pass

    if hasattr(torch, 'xpu') and torch.xpu.is_available():
        device = torch.device("xpu")
        print(f"[DEVICE] 使用 Intel XPU: {torch.xpu.get_device_name(0)}")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[DEVICE] 使用 CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("[DEVICE] 警告: 无 GPU 可用，使用 CPU（速度会很慢）")
    return device

DEVICE = get_device()
IS_XPU = (DEVICE.type == "xpu")
IS_CUDA = (DEVICE.type == "cuda")

DATASET_DIR = os.path.join(ROOT_DIR, "datasets", "medical-pills")

# 模型权重路径
MODELS_DIR = os.path.join(ROOT_DIR, "models")
YOLO_PRETRAINED = "yolo11n.pt"          # YOLO11 预训练权重
YOLO_TRAINED = os.path.join(MODELS_DIR, "yolo11n_pills_best.pt")  # 训练后保存路径
SAM2_MODEL = "sam2_s.pt"                # SAM2 模型

# 输出目录
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
UPLOAD_DIR = os.path.join(ROOT_DIR, "uploads")

# 确保必要目录存在
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UPLOAD_DIR, exist_ok=True)

# YOLO11 训练参数
YOLO_TRAIN_CONFIG = {
    "data": os.path.join(ROOT_DIR, "medical-pills.yaml"),  # 数据集 YAML
    "epochs": 100,
    "imgsz": 640,
    "batch": 8,              # XPU 16GB 显存，保守设置
    "device": str(DEVICE),   # "xpu" 或 "cuda"
    "workers": 0,            # Windows + XPU 下必须为 0
    "patience": 20,          # 早停
    "save": True,
    "save_period": 10,
    "exist_ok": True,
    "pretrained": True,
    "optimizer": "auto",
    "verbose": True,
    "project": os.path.join(ROOT_DIR, "runs", "train"),
    "name": "yolo11_pills",
}

# SAM2 推理参数
SAM2_CONFIG = {
    "device": str(DEVICE),
    "conf": 0.5,             # 置信度阈值
}

# 可视化参数
VIS_CONFIG = {
    # 检测图
    "detection_box_color": (0, 140, 255),    # 框颜色 BGR
    # 分割图（SAM2 掩码 + 轮廓 + 检测框）
    "segmentation_box_color": (0, 0, 255),   # 框/轮廓颜色 BGR
    "segmentation_mask_color": (0, 0, 255),  # 掩码填充颜色
    "segmentation_mask_alpha": 0.4,          # 掩码透明度
    # 通用
    "detection_mask_alpha": 0.4,             # 保留兼容
    "line_thickness": 2,
    "font_scale": 0.6,
}