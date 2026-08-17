# Flask Web 应用
import sys
import os
import io
import base64
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_file

sys.path.insert(0, str(Path(__file__).parent))

from config import (
    DEVICE, YOLO_TRAINED, SAM2_MODEL, SAM2_CONFIG,
    UPLOAD_DIR, OUTPUT_DIR, ROOT_DIR,
)
from utils.xpu_patch import apply_xpu_patches, print_device_info

# Flask 应用初始化
app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB 上传限制

# 确保上传目录存在
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 加载模型（全局单例）
print("=" * 60)
print("初始化 YOLO11 + SAM2 Web 服务")
print("=" * 60)
print_device_info(DEVICE)
apply_xpu_patches(DEVICE)

from ultralytics import YOLO, SAM
from inference_pipeline import draw_results

print(f"\n加载 YOLO11 模型: {YOLO_TRAINED}")
if not Path(YOLO_TRAINED).exists():
    print(f"[警告] YOLO11 模型未找到: {YOLO_TRAINED}")
    yolo_model = None
else:
    yolo_model = YOLO(YOLO_TRAINED)

print(f"加载 SAM2 模型: {SAM2_MODEL}")
try:
    sam_model = SAM(SAM2_MODEL)
except Exception as e:
    print(f"[警告] SAM2 模型加载失败: {e}")
    sam_model = None

print("模型加载完成！")
print("=" * 60)

# 路由
@app.route('/')
def index():
    # 主页
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # 上传图片并推理
    if yolo_model is None or sam_model is None:
        return jsonify({'error': '模型未加载，请先训练 YOLO11 并确保 SAM2 可用'}), 500

    if 'image' not in request.files:
        return jsonify({'error': '未上传图片'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': '文件名为空'}), 400

    # 读取图片
    img_bytes = file.read()
    img_array = np.frombuffer(img_bytes, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if image is None:
        return jsonify({'error': '无法解码图片'}), 400

    # 保存原始上传图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    upload_path = Path(UPLOAD_DIR) / f"{timestamp}_{file.filename}"
    cv2.imwrite(str(upload_path), image)

    # 推理
    try:
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

        # 编码为 base64
        def img_to_base64(img):
            _, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 90])
            return base64.b64encode(buffer).decode('utf-8')

        result = {
            'success': True,
            'detections': num_detections,
            'detection_image': f"data:image/jpeg;base64,{img_to_base64(det_img)}",
            'segmentation_image': f"data:image/jpeg;base64,{img_to_base64(seg_img)}",
            'original_filename': file.filename,
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': f'推理失败: {str(e)}'}), 500


@app.route('/health')
def health():
    # 健康检查
    return jsonify({
        'status': 'ok',
        'yolo_loaded': yolo_model is not None,
        'sam_loaded': sam_model is not None,
        'device': str(DEVICE),
    })


if __name__ == '__main__':
    print("\n启动 Web 服务...")
    print("请在浏览器打开: http://localhost:5000")
    app.run(host='0.0.0.0', port=5000, debug=False)