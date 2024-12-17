import torch
import numpy as np
import cv2
from torchvision.ops import nms

import torch
from ultralytics import YOLO

class YoloV10Detector:
    def __init__(self, model_path, img_size=640, conf_thresh=0.3, iou_thresh=0.45, device=None):
        """
        Khởi tạo YOLOv10n Detector.
        """
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load YOLOv10n model
        print("Loading YOLOv10n model from:", model_path)
        self.model = YOLO(model_path)  # Sử dụng YOLO từ Ultralytics
        self.model.to(self.device)
        print("Model loaded successfully.")

    def predict(self, image):
        """
        Dự đoán các bounding boxes cho hình ảnh đầu vào.
        """
        print("Running inference...")
        results = self.model.predict(
            source=image,
            imgsz=self.img_size,
            conf=self.conf_thresh,
            iou=self.iou_thresh,
            device=self.device,
            verbose=False
        )
        return results[0]  # Trả về kết quả đầu tiên

    def postprocess(self, preds, img_shape):
        """
        Xử lý kết quả dự đoán.
        :param preds: Kết quả từ model (raw output).
        :param img_shape: Kích thước ảnh gốc (H, W).
        :return: Danh sách các bounding boxes [(x1, y1, x2, y2, conf)].
        """
        boxes = []
        confidences = []

        # Giả sử preds đã là tensor [num_boxes, 6] (x1, y1, x2, y2, conf, class)
        preds = preds[0]  # Bỏ batch dimension

        for pred in preds:
            x1, y1, x2, y2, conf = pred[:5]
            if conf > self.conf_thresh:
                boxes.append([x1, y1, x2, y2])
                confidences.append(conf)

        # Áp dụng NMS để loại bỏ các box trùng
        if len(boxes) > 0:
            boxes = torch.tensor(boxes, device=self.device)
            confidences = torch.tensor(confidences, device=self.device)
            keep = nms(boxes, confidences, self.iou_thresh)
            boxes = boxes[keep].cpu().numpy()
            confidences = confidences[keep].cpu().numpy()

            # Scale boxes về kích thước ảnh gốc
            h, w = img_shape
            scale_w, scale_h = w / self.img_size, h / self.img_size
            final_boxes = []
            for box, conf in zip(boxes, confidences):
                x1, y1, x2, y2 = box
                x1 = int(x1 * scale_w)
                y1 = int(y1 * scale_h)
                x2 = int(x2 * scale_w)
                y2 = int(y2 * scale_h)
                final_boxes.append((x1, y1, x2, y2, float(conf)))
            return final_boxes
        return []

    def predict(self, img):
        """
        Chạy dự đoán trên ảnh.
        :param img: Ảnh numpy (H x W x C).
        :return: Danh sách bounding boxes [(x1, y1, x2, y2, conf)].
        """
        img_tensor = self.preprocess(img)
        with torch.no_grad():
            preds = self.model(img_tensor)  # Dự đoán
        results = self.postprocess(preds, img.shape[:2])
        return results

    def dump_cache(self):
        """
        Làm sạch bộ nhớ cache (nếu cần).
        """
        torch.cuda.empty_cache()
