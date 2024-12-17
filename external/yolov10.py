import torch
import numpy as np
import cv2
from torchvision.ops import nms
from ultralytics import YOLO

class YoloV10Detector:
    def __init__(self, model_path, img_size=640, conf_thresh=0.3, iou_thresh=0.45, device=None):
        """
        Khởi tạo YOLOv10 Detector.
        """
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Load YOLOv10 model
        print("Loading YOLOv10x model from:", model_path)
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
            img = img.cpu().numpy()  # Chuyển sang HWC (Height-Width-Channel)

        print(img.shape)
        print(self.img_size)
        # Kiểm tra lại kiểu dữ liệu
        
        img_resized = cv2.resize(img, (self.img_size[1], self.img_size[0]))  # Resize về kích thước yêu cầu

        # Chuyển ảnh sang định dạng [C, H, W] và normalize
        img_resized = img_resized[:, :, ::-1]  # BGR to RGB
        img_resized = img_resized.copy()

        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0  # Normalize [0,1]
        img_tensor = img_tensor.unsqueeze(0)  # Thêm batch dimension [1, C, H, W]
        img_tensor = img_tensor.to(self.device)
        return img_tensor

    def postprocess(self, results, img_shape):
        """
        Xử lý kết quả dự đoán.
        :param results: Kết quả từ model (raw output).
        :param img_shape: Kích thước ảnh gốc (H, W).
        :return: Tensor các bounding boxes [(xmin, ymin, xmax, ymax, conf)].
        """
        boxes = []
        confidences = []
        labels = []

        # Giả sử preds đã là tensor [num_boxes, 6] (x, y, w, h, conf, class)
        for result in results:
            boxes.extend(result.boxes.xywh.cpu().numpy())  # Tọa độ bounding box
            confidences.extend(result.boxes.conf.cpu().numpy())  # Độ tin cậy
            labels.extend(result.boxes.cls.cpu().numpy())  # Lớp của đối tượng
        
        # Áp dụng NMS để loại bỏ các box trùng
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        labels = np.array(labels)

        # Chuyển bounding boxes về kích thước ảnh gốc
        print(img_shape) 
        h, w = img_shape
        scale_w, scale_h = float(w / self.img_size[1]), float(h / self.img_size[0])
        boxes[:, [0, 2]] *= scale_w
        boxes[:, [1, 3]] *= scale_h

        # Chuyển đổi từ [x_center, y_center, w, h] sang [xmin, ymin, xmax, ymax]
        boxes[:, 0] -= boxes[:, 2] / 2.0  # xmin
        boxes[:, 1] -= boxes[:, 3] / 2.0  # ymin
        boxes[:, 2] += boxes[:, 0]  # xmax = xmin + w
        boxes[:, 3] += boxes[:, 1]  # ymax = ymin + h

        # Đưa kết quả về dạng [xmin, ymin, xmax, ymax, conf]
        results = []
        for box, conf, label in zip(boxes, confidences, labels):
            results.append([float(box[0]), float(box[1]), float(box[2]), float(box[3]), float(conf)])

        return results


    def predict(self, img):
        """
        Chạy dự đoán trên ảnh.
        :param img: Ảnh numpy (H x W x C).
        :return: Tensor các bounding boxes [(x1, y1, x2, y2, conf)].
        """
        img_tensor = self.preprocess(img)
        with torch.no_grad():
            # Dự đoán và nhận kết quả
            results = self.model(img_tensor)
        
        # Lọc kết quả và post-process
        final_results = self.postprocess(results, img.shape[:2])
        return final_results

    def dump_cache(self):
        """
        Làm sạch bộ nhớ cache (nếu cần).
        """
        torch.cuda.empty_cache()
