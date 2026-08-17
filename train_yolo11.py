# YOLO11 训练脚本
# 在 Medical Pills 数据集上训练 YOLO11 检测模型
import sys
import torch
from pathlib import Path

# 添加项目根目录到 path
sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEVICE, IS_XPU, YOLO_TRAIN_CONFIG, YOLO_PRETRAINED, YOLO_TRAINED,
    MODELS_DIR, ROOT_DIR,
)
from utils.xpu_patch import apply_xpu_patches, print_device_info


def main():
    print("=" * 60)
    print("YOLO11 训练 - Medical Pills 数据集")
    print("=" * 60)

    # 1. 打印设备信息
    print_device_info(DEVICE)

    # 2. 应用 XPU 兼容性补丁
    apply_xpu_patches(DEVICE)

    # 3. 导入 ultralytics
    from ultralytics import YOLO

    # 4. 检查数据集路径
    data_yaml = YOLO_TRAIN_CONFIG["data"]
    if not Path(data_yaml).exists():
        print(f"[错误] 数据集配置文件不存在: {data_yaml}")
        print("请先下载 Medical Pills 数据集并修改 medical-pills.yaml 中的 path")
        sys.exit(1)

    print(f"\n数据集配置: {data_yaml}")
    print(f"训练参数:")
    for k, v in YOLO_TRAIN_CONFIG.items():
        if k != "data":
            print(f"  {k}: {v}")

    # 5. XPU 训练特殊设置
    train_kwargs = YOLO_TRAIN_CONFIG.copy()

    if IS_XPU:
        # XPU 禁用 AMP
        train_kwargs["amp"] = False
        print("\n[XPU] AMP 已禁用，使用 FP32 训练")
        print("[XPU] num_workers=0 (Windows + XPU 兼容)")
    else:
        train_kwargs["amp"] = True

    # 6. 加载模型并训练
    print(f"\n加载预训练模型: {YOLO_PRETRAINED}")
    model = YOLO(YOLO_PRETRAINED)

    print("\n开始训练...")
    print("-" * 60)

    try:
        results = model.train(**train_kwargs)

        # 7. 保存最佳模型到指定路径
        best_pt = Path(results.save_dir) / "weights" / "best.pt"
        if best_pt.exists():
            import shutil
            shutil.copy(best_pt, YOLO_TRAINED)
            print(f"\n最佳模型已保存到: {YOLO_TRAINED}")

        print("\n" + "=" * 60)
        print("训练完成！")
        print(f"训练结果目录: {results.save_dir}")
        print(f"最佳模型路径: {YOLO_TRAINED}")
        print("=" * 60)

    except KeyboardInterrupt:
        print("\n训练被用户中断")
    except Exception as e:
        print(f"\n训练出错: {e}")
        raise


if __name__ == "__main__":
    main()