import cv2
import time
import os
from external.adaptors import detector
from tracker.boost_track import BoostTrack
import utils
from default_settings import GeneralSettings, get_detector_path_and_im_size
import torch
import cv2

def preprocess_image(image, input_size):
    h, w = input_size
    image = cv2.resize(image, (w, h))
    image = image.astype('float32') / 255.0  # Normalize
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

def process_video(video_path, output_path, model_path="external/weights/bytetrack_x_mot17.pth.tar", detector_model="yolox", dataset="mot17"):
    # Initialize detector
    det = detector.Detector(detector_model, model_path, dataset)
    print(next(det.model.parameters()).device)  # Truy cập thiết bị của tham số đầu tiên trong mô hình

    # Initialize video reader and writer
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for saving MP4 videos
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Initialize tracker
    tracker = BoostTrack(video_name="MOT_Pipeline")
    frame_count = 0

    print("Starting MOT pipeline...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_size = (800, 1440)           # Kích thước đầu vào yêu cầu bởi mô hình
        preprocessed = preprocess_image(frame_rgb, input_size)
        preprocessed = preprocessed.to(next(det.model.parameters()).device)  # Chuyển dữ liệu vào cùng thiết bị với mô hình

        # Object detection
        pred = det(preprocessed, tag=f"frame_{frame_count}")
        if pred is None:  # No detection
            out.write(frame)
            continue

        # Update tracker
        targets = tracker.update(pred, preprocessed, frame, tag=f"frame_{frame_count}")

        # Filter and draw bounding boxes
        tlwhs, ids, confs = utils.filter_targets(targets, 1.6, 10)  # Example thresholds
        for tlwh, track_id, conf in zip(tlwhs, ids, confs):
            x1, y1, w, h = map(int, tlwh)
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)  # Green boxes
            label = f"ID: {track_id} ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        print(f"Processed frame {frame_count}\r", end="")

    cap.release()
    out.release()
    print("\nMOT pipeline completed!")
    print(f"Output saved to: {output_path}")

# Example usage
input_video = "/kaggle/input/mot17-videos/MOT17-01-FRCNN-raw.mp4"
output_video = "/kaggle/working/MOT17-01-FRCNN-raw.mp4"
process_video(input_video, output_video)