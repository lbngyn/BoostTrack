import torch
import numpy as np
import cv2
from torchvision.ops import nms
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

    def preprocess(self, img):
        """
        Chuyển đổi ảnh numpy thành tensor phù hợp để model dự đoán.
        :param img: Ảnh numpy (H x W x C).
        :return: Tensor ảnh đã chuẩn hóa.
        """
        # Nếu img là Tensor PyTorch
        if torch.is_tensor(img):
            if img.ndim == 4:  # Loại bỏ batch dimension
                img = img.squeeze(0)
            img = img.permute(1, 2, 0).cpu().numpy()  # Chuyển sang HWC (Height-Width-Channel)

        # Kiểm tra lại kiểu dữ liệu
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Input img must be a numpy array after preprocessing, got {type(img)}")
        
        # Kiểm tra nếu img_size là một tuple (height, width)
        if isinstance(self.img_size, tuple) and len(self.img_size) == 2:
            # Resize ảnh về kích thước model yêu cầu (width, height)
            img_resized = cv2.resize(img, (self.img_size[1], self.img_size[0]))  # (width, height)
        else:
            raise ValueError(f"Expected img_size to be a tuple (height, width), got {self.img_size}")

        # Chuyển ảnh sang định dạng [C, H, W] và normalize
        img_resized = img_resized[:, :, ::-1]  # BGR to RGB
        img_resized = img_resized.copy()

        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0  # Normalize [0,1]
        img_tensor = img_tensor.unsqueeze(0)  # Thêm batch dimension [1, C, H, W]
        img_tensor = img_tensor.to(self.device)
        return img_tensor

    def postprocess(self, preds, img_shape):
        """
        Xử lý kết quả dự đoán.
        :param preds: Kết quả từ model (raw output).
        :param img_shape: Kích thước ảnh gốc (H, W).
        :return: Tensor các bounding boxes [(x1, y1, x2, y2, conf)].
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
            boxes = boxes[keep].cpu()
            confidences = confidences[keep].cpu()

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
                final_boxes.append([x1, y1, x2, y2, float(conf)])

            # Chuyển về tensor
            return torch.tensor(final_boxes, device=self.device)
        return torch.tensor([]).to(self.device)

    def predict(self, img):
        """
        Chạy dự đoán trên ảnh.
        :param img: Ảnh numpy (H x W x C).
        :return: Tensor các bounding boxes [(x1, y1, x2, y2, conf)].
        """
        print(f"Type of img: {type(img)}")
        print(f"Shape of img: {getattr(img, 'shape', 'Not available')}")

        img_tensor = self.preprocess(img)
        with torch.no_grad():
            preds = self.model(img_tensor)  # Dự đoán
        
        print(len(preds))
        results = self.postprocess(preds, img.shape[:2])
        return results

    def dump_cache(self):
        """
        Làm sạch bộ nhớ cache (nếu cần).
        """
        torch.cuda.empty_cache()
