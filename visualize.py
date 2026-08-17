# 可视化脚本
# 读取训练好的模型，对验证集或指定图像进行推理并可视化结果
import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEVICE, YOLO_TRAINED, SAM2_MODEL, SAM2_CONFIG,
    OUTPUT_DIR, VIS_CONFIG, DATASET_DIR, ROOT_DIR,
)
from utils.xpu_patch import apply_xpu_patches, print_device_info


def visualize_single(image_path, yolo_model, sam_model, output_dir, show=False):
    # 对单张图像进行推理并可视化
    from inference_pipeline import draw_results

    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[警告] 无法读取图像: {image_path}")
        return None

    # YOLO11 检测
    yolo_results = yolo_model(image, conf=SAM2_CONFIG["conf"],
                              device=str(DEVICE), verbose=False)

    num_detections = 0
    if yolo_results[0].boxes is not None:
        num_detections = len(yolo_results[0].boxes)

    # SAM2 分割
    sam_results = None
    if yolo_results[0].boxes is not None and len(yolo_results[0].boxes) > 0:
        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        sam_results = sam_model(image, bboxes=boxes.tolist(),
                                device=str(DEVICE), verbose=False)

    # 可视化
    det_img, seg_img = draw_results(image, yolo_results, sam_results)

    # 保存
    img_name = Path(image_path).stem
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    det_path = out_dir / f"{img_name}_det.jpg"
    seg_path = out_dir / f"{img_name}_seg.jpg"
    cv2.imwrite(str(det_path), det_img)
    cv2.imwrite(str(seg_path), seg_img)

    print(f"  [{img_name}] 检测: {num_detections} 个药丸 → {det_path.name}, {seg_path.name}")

    if show:
        cv2.imshow("Detection (Orange)", det_img)
        cv2.imshow("Segmentation (Red)", seg_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return det_img, seg_img, num_detections


def visualize_val_set(yolo_model, sam_model, num_images=5, output_dir=None):
    # 可视化验证集中的图像
    out_dir = Path(output_dir) if output_dir else Path(OUTPUT_DIR) / "val_vis"
    out_dir.mkdir(parents=True, exist_ok=True)

    val_img_dir = Path(DATASET_DIR) / "images" / "val"
    if not val_img_dir.exists():
        print(f"[错误] 验证集目录不存在: {val_img_dir}")
        print("请确认数据集已下载到正确位置")
        return

    img_files = sorted(val_img_dir.glob("*"))[:num_images]
    print(f"可视化验证集: {len(img_files)} 张图像\n")

    total_detections = 0
    for img_file in img_files:
        _, _, count = visualize_single(img_file, yolo_model, sam_model, out_dir)
        total_detections += count

    print(f"\n总计检测到 {total_detections} 个药丸")


def main():
    parser = argparse.ArgumentParser(description="YOLO11 + SAM2 可视化脚本")
    parser.add_argument("--image", type=str, help="单张图像路径")
    parser.add_argument("--dir", type=str, help="图像目录")
    parser.add_argument("--val", action="store_true", help="可视化验证集")
    parser.add_argument("--num", type=int, default=5, help="验证集可视化数量")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    parser.add_argument("--show", action="store_true", help="显示图像窗口")
    parser.add_argument("--yolo", type=str, default=None, help="YOLO11 模型路径")
    parser.add_argument("--conf", type=float, default=None, help="置信度阈值")
    args = parser.parse_args()

    # 初始化
    print("=" * 60)
    print("YOLO11 + SAM2 可视化")
    print("=" * 60)
    print_device_info(DEVICE)
    apply_xpu_patches(DEVICE)

    from ultralytics import YOLO, SAM

    # 加载模型
    yolo_path = args.yolo or YOLO_TRAINED
    if not Path(yolo_path).exists():
        print(f"[错误] YOLO11 模型未找到: {yolo_path}")
        print("请先运行 train_yolo11.py 训练模型")
        sys.exit(1)

    print(f"\n加载 YOLO11: {yolo_path}")
    yolo_model = YOLO(yolo_path)
    print(f"加载 SAM2: {SAM2_MODEL}")
    sam_model = SAM(SAM2_MODEL)

    output_dir = args.output or OUTPUT_DIR

    if args.image:
        visualize_single(args.image, yolo_model, sam_model, output_dir, show=args.show)
    elif args.dir:
        img_dir = Path(args.dir)
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        img_files = [f for f in img_dir.iterdir() if f.suffix.lower() in img_exts]
        print(f"处理 {len(img_files)} 张图像\n")
        for img_file in img_files:
            visualize_single(img_file, yolo_model, sam_model, output_dir, show=args.show)
    elif args.val:
        visualize_val_set(yolo_model, sam_model, num_images=args.num, output_dir=output_dir)
    else:
        parser.print_help()
        print("\n示例:")
        print("  python visualize.py --val --num 5")
        print("  python visualize.py --image test.jpg --show")
        print("  python visualize.py --dir my_images/")


if __name__ == "__main__":
    main()