import torch
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from external.yolov10 import YoloV10Detector  # Module detector mới của bạn

def load_image(image_path):
    """
    Load image từ file và chuyển đổi thành định dạng numpy array.
    :param image_path: Đường dẫn tới file ảnh.
    :return: Ảnh dưới dạng numpy array.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {image_path}")
    return img

def display_results(img, results):
    """
    Hiển thị kết quả phát hiện đối tượng.
    :param img: Ảnh numpy (H x W x C).
    :param results: Dự đoán bounding boxes [(x1, y1, x2, y2, conf)].
    """
    for box in results:
        x1, y1, x2, y2, conf = box
        print(box)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{conf:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Chuyển đổi BGR sang RGB để sử dụng với matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')  # Tắt trục
    plt.show()

if __name__ == "__main__":
    model_path = '/kaggle/input/yolov10x/other/default/1/yolov10x.pt'  # Thay bằng đường dẫn đến model YOLOv10n của bạn
    image_path = '/kaggle/input/mot17-converted-coco/MOT17/train/MOT17-02-FRCNN/img1/000302.jpg'  # Thay bằng đường dẫn đến ảnh bạn muốn kiểm tra

    # Tạo đối tượng detector
    detector = YoloV10Detector(model_path=model_path)

    # Load ảnh
    img = load_image(image_path)
    print(img.shape)

    # Dự đoán
    results = detector.predict(img)

    # Hiển thị kết quả
    display_results(img, results)
