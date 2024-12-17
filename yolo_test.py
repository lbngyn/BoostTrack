import torch
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

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
        if not isinstance(img, np.ndarray):
            raise TypeError(f"Input img must be a numpy array, got {type(img)}")

        # Resize ảnh
        img_resized = cv2.resize(img, (self.img_size, self.img_size))  # Resize về kích thước yêu cầu
        img_resized = img_resized[:, :, ::-1]  # BGR to RGB
        img_resized = img_resized.copy()

        # Chuyển ảnh sang tensor
        img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0  # Normalize [0,1]
        img_tensor = img_tensor.unsqueeze(0)  # Thêm batch dimension [1, C, H, W]
        img_tensor = img_tensor.to(self.device)
        return img_tensor

    def postprocess(self, results, img_shape):
        """
        Xử lý kết quả dự đoán.
        :param results: Kết quả từ model (raw output).
        :param img_shape: Kích thước ảnh gốc (H, W).
        :return: Tensor các bounding boxes [(x1, y1, x2, y2, conf)].
        """
        boxes = []
        confidences = []
        labels = []
        print(labels)

        # Giả sử preds đã là tensor [num_boxes, 6] (x1, y1, x2, y2, conf, class)
        for result in results:
            boxes.extend(result.boxes.xywh.cpu().numpy())  # Tọa độ bounding box
            confidences.extend(result.boxes.conf.cpu().numpy())  # Độ tin cậy
            labels.extend(result.boxes.cls.cpu().numpy())  # Lớp của đối tượng
        
        print(labels)

        # Áp dụng NMS để loại bỏ các box trùng
        boxes = np.array(boxes)
        confidences = np.array(confidences)
        labels = np.array(labels)

        # Chuyển bounding boxes về kích thước ảnh gốc
        h, w = img_shape
        scale_w, scale_h = w / self.img_size, h / self.img_size
        boxes[:, [0, 2]] *= scale_w
        boxes[:, [1, 3]] *= scale_h

        # Lọc theo độ tin cậy
        mask = confidences > self.conf_thresh
        boxes = boxes[mask]
        confidences = confidences[mask]
        labels = labels[mask]

        # Đưa kết quả về dạng [x1, y1, x2, y2, conf]
        results = []
        for box, conf, label in zip(boxes, confidences, labels):
            results.append([int(box[0]), int(box[1]), int(box[2]), int(box[3]), float(conf)])

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
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(img, f'{conf:.2f}', (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    # Chuyển đổi BGR sang RGB để sử dụng với matplotlib
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_rgb)
    plt.axis('off')  # Tắt trục
    plt.show()

if __name__ == "__main__":
    model_path = '/kaggle/input/yolov10x/other/default/1/yolov10x.pt'  # Thay bằng đường dẫn đến model YOLOv10n của bạn
    image_path = '/kaggle/input/mot17-converted-coco/MOT17/train/MOT17-02-DPM/img1/000001.jpg'  # Thay bằng đường dẫn đến ảnh bạn muốn kiểm tra

    # Tạo đối tượng detector
    detector = YoloV10Detector(model_path=model_path)

    # Load ảnh
    img = load_image(image_path)

    # Dự đoán
    results = detector.predict(img)

    # Hiển thị kết quả
    display_results(img, results)
