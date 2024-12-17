import torch
import numpy as np
import cv2
from torchvision.ops import nms

class YoloV10Detector:
    def __init__(self, model_path, img_size=640, conf_thresh=0.3, iou_thresh=0.45, device=None):
        """
        Khởi tạo YOLOv10n Detector.
        :param model_path: Đường dẫn đến file model YOLOv10n (.pt).
        :param img_size: Kích thước ảnh đầu vào.
        :param conf_thresh: Ngưỡng tin cậy cho detection.
        :param iou_thresh: Ngưỡng NMS (non-maximum suppression).
        :param device: CPU hoặc GPU.
        """
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device if device else (torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
        # Load model YOLOv10n
        print("Loading YOLOv10n model from:", model_path)
        self.model = torch.load(model_path, map_location=self.device)
        self.model.eval()
        print("Model loaded successfully.")

    def preprocess(self, img):
        """
        Tiền xử lý ảnh đầu vào.
        :param img: Ảnh numpy (H x W x C).
        :return: Tensor ảnh đã xử lý.
        """
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        img_resized = img_resized / 255.0  # Chuẩn hóa
        img_resized = np.transpose(img_resized, (2, 0, 1))  # HWC -> CHW
        img_tensor = torch.tensor(img_resized, dtype=torch.float).unsqueeze(0).to(self.device)
        return img_tensor

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
