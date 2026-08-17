# YOLO11 + SAM2 联合推理管线
# YOLO11 检测药丸 → 检测框作为 SAM2 的框提示 → SAM2 实例化分割
import argparse
import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEVICE, IS_XPU, YOLO_TRAINED, SAM2_MODEL, SAM2_CONFIG,
    OUTPUT_DIR, VIS_CONFIG,
)
from utils.xpu_patch import apply_xpu_patches, print_device_info, get_gpu_memory_info

# 绘制检测结果和分割结果
def draw_results(image, yolo_results, sam_results, vis_mode="both"):
    det_img = image.copy()
    seg_img = image.copy()

    if yolo_results is None or len(yolo_results) == 0:
        return det_img, seg_img

    boxes = yolo_results[0].boxes
    if boxes is None or len(boxes) == 0:
        return det_img, seg_img

    # 获取 SAM2 掩码
    masks = None
    if sam_results is not None and len(sam_results) > 0:
        if hasattr(sam_results[0], 'masks') and sam_results[0].masks is not None:
            masks = sam_results[0].masks.data

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        label = f"pill {conf:.2f}"

        # 检测图：仅画 YOLO11 检测框
        cv2.rectangle(det_img, (x1, y1), (x2, y2),
                      VIS_CONFIG["detection_box_color"], VIS_CONFIG["line_thickness"])
        cv2.putText(det_img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, VIS_CONFIG["font_scale"],
                    VIS_CONFIG["detection_box_color"], VIS_CONFIG["line_thickness"])

        # 分割图：画 SAM2 掩码 + 轮廓 + 检测框
        if masks is not None and i < len(masks):
            mask = masks[i].cpu().numpy().astype(np.uint8)
            if mask.shape[:2] != seg_img.shape[:2]:
                mask = cv2.resize(mask, (seg_img.shape[1], seg_img.shape[0]))

            # 掩码覆盖
            colored_mask = np.zeros_like(seg_img)
            colored_mask[mask > 0] = VIS_CONFIG["segmentation_mask_color"]
            seg_img = cv2.addWeighted(seg_img, 1.0, colored_mask,
                                      VIS_CONFIG["segmentation_mask_alpha"], 0)

            # 掩码轮廓
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(seg_img, contours, -1,
                             VIS_CONFIG["segmentation_box_color"],
                             VIS_CONFIG["line_thickness"])

        # 分割图上也画检测框
        cv2.rectangle(seg_img, (x1, y1), (x2, y2),
                      VIS_CONFIG["segmentation_box_color"], VIS_CONFIG["line_thickness"])
        cv2.putText(seg_img, label, (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, VIS_CONFIG["font_scale"],
                    VIS_CONFIG["segmentation_box_color"], VIS_CONFIG["line_thickness"])

    # 添加标题
    det_img = _add_title(det_img, "Detection | YOLO11")
    seg_img = _add_title(seg_img, "Segmentation | SAM2")

    return det_img, seg_img

# 添加标题
def _add_title(img, title):
    h, w = img.shape[:2]
    title_bar = np.zeros((40, w, 3), dtype=np.uint8)
    title_bar[:] = (50, 50, 50)
    cv2.putText(title_bar, title, (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    return np.vstack([title_bar, img])

# 运行推理
def run_inference(image_path, yolo_model_path=None, sam2_model_name=None, 
                  conf_threshold=None, output_dir=None):

    # 初始化
    yolo_path = yolo_model_path or YOLO_TRAINED
    sam2_name = sam2_model_name or SAM2_MODEL
    conf = conf_threshold or SAM2_CONFIG["conf"]
    out_dir = Path(output_dir) if output_dir else Path(OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    print_device_info(DEVICE)
    apply_xpu_patches(DEVICE)

    from ultralytics import YOLO, SAM

    # 加载模型
    print(f"\n加载 YOLO11 模型: {yolo_path}")
    if not Path(yolo_path).exists():
        print(f"[错误] YOLO11 模型未找到: {yolo_path}")
        sys.exit(1)

    yolo_model = YOLO(yolo_path)
    print(f"加载 SAM2 模型: {sam2_name}")
    sam_model = SAM(sam2_name)

    # 读取图像
    print(f"\n处理图像: {image_path}")
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"[错误] 无法读取图像: {image_path}")
        sys.exit(1)
    print(f"图像尺寸: {image.shape[1]}x{image.shape[0]}")

    # Step 1: YOLO11 检测
    print("\n[Step 1] YOLO11 检测...")
    yolo_results = yolo_model(image, conf=conf, device=str(DEVICE), verbose=False)

    if yolo_results[0].boxes is None or len(yolo_results[0].boxes) == 0:
        print("未检测到任何药丸！")
        return image, image, 0

    num_detections = len(yolo_results[0].boxes)
    print(f"检测到 {num_detections} 个药丸")

    # Step 2: 提取检测框作为 SAM2 框提示
    boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
    print(f"[Step 2] SAM2 分割（使用 {len(boxes)} 个检测框作为提示）...")

    sam_results = sam_model(image, bboxes=boxes.tolist(), device=str(DEVICE), verbose=False)

    # Step 3: 可视化
    print("[Step 3] 生成可视化结果...")
    det_img, seg_img = draw_results(image, yolo_results, sam_results)

    # 保存结果
    img_name = Path(image_path).stem
    det_path = out_dir / f"{img_name}_detection.jpg"
    seg_path = out_dir / f"{img_name}_segmentation.jpg"
    combined_path = out_dir / f"{img_name}_combined.jpg"

    cv2.imwrite(str(det_path), det_img)
    cv2.imwrite(str(seg_path), seg_img)
    print(f"检测结果: {det_path}")
    print(f"分割结果: {seg_path}")

    # 合并图（左右拼接）
    h = max(det_img.shape[0], seg_img.shape[0])
    det_padded = np.zeros((h, det_img.shape[1], 3), dtype=np.uint8)
    seg_padded = np.zeros((h, seg_img.shape[1], 3), dtype=np.uint8)
    det_padded[:det_img.shape[0]] = det_img
    seg_padded[:seg_img.shape[0]] = seg_img
    combined = np.hstack([det_padded, seg_padded])
    cv2.imwrite(str(combined_path), combined)
    print(f"合并结果: {combined_path}")

    print(get_gpu_memory_info(DEVICE))
    print("推理完成！")

    return det_img, seg_img, num_detections


def main():
    parser = argparse.ArgumentParser(description="YOLO11 + SAM2 联合推理")
    parser.add_argument("--image", type=str, help="单张图像路径")
    parser.add_argument("--dir", type=str, help="图像目录（批量推理）")
    parser.add_argument("--output", type=str, default=None, help="输出目录")
    parser.add_argument("--yolo", type=str, default=None, help="YOLO11 模型路径")
    parser.add_argument("--sam2", type=str, default=None, help="SAM2 模型名称")
    parser.add_argument("--conf", type=float, default=None, help="置信度阈值")
    args = parser.parse_args()

    if args.image:
        run_inference(
            image_path=args.image,
            yolo_model_path=args.yolo,
            sam2_model_name=args.sam2,
            conf_threshold=args.conf,
            output_dir=args.output,
        )
    elif args.dir:
        img_dir = Path(args.dir)
        img_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        img_files = [f for f in img_dir.iterdir() if f.suffix.lower() in img_exts]
        print(f"找到 {len(img_files)} 张图像\n")
        for img_file in img_files:
            print(f"\n{'='*40}")
            run_inference(
                image_path=str(img_file),
                yolo_model_path=args.yolo,
                sam2_model_name=args.sam2,
                conf_threshold=args.conf,
                output_dir=args.output,
            )
    else:
        parser.print_help()
        print("\n示例: python inference_pipeline.py --image test.jpg")


if __name__ == "__main__":
    main()