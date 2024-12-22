import argparse
import cv2
from tracker.boost_track import BoostTrack
import utils
from default_settings import GeneralSettings, get_detector_path_and_im_size
import torch
from external.yolov10 import YoloV10Detector

def preprocess_image(image, input_size):
    h, w = input_size
    image = cv2.resize(image, (w, h))
    image = image.astype('float32') / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

def process_video(video_path, output_path, model_path, det_classes):
    size = (800, 1440)
    det = YoloV10Detector(model_path=model_path, img_size=size)
    print(next(det.model.parameters()).device)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracker = BoostTrack(video_name="MOT_Pipeline")
    frame_count = 0

    print("Starting MOT pipeline...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).unsqueeze(0).float()
        frame_rgb_tensor = frame_rgb_tensor.to(next(det.model.parameters()).device)

        pred = det.predict(frame_rgb, det_classes)
        pred = torch.tensor(pred)

        print(pred)
        if pred is None or len(pred) == 0:
            out.write(frame)
            continue

        targets = tracker.update(pred, frame_rgb_tensor, frame_rgb, tag=f"frame_{frame_count}")
        tlwhs, ids, confs = utils.filter_targets(targets, 1.6, 10)
        for tlwh, track_id, conf in zip(tlwhs, ids, confs):
            x1, y1, w, h = map(int, tlwh)
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
            label = f"ID: {track_id} ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        out.write(frame)
        print(f"Processed frame {frame_count}\r", end="")

    cap.release()
    out.release()
    print("\nMOT pipeline completed!")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-Object Tracking Pipeline")
    parser.add_argument("--video", type=str, default="/kaggle/input/mot17-videos/MOT17-02-FRCNN-raw.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default="/kaggle/working/MOT17-02-FRCNN-raw.mp4", help="Path to save output video")
    parser.add_argument("--model", type=str, default="/kaggle/input/yolov10x/other/default/1/yolov10x.pt", help="Path to YOLOv10x model")
    parser.add_argument("--det_classes", nargs="+", type=int, default=[0], help="Detection classes (default: [0])")

    args = parser.parse_args()

    process_video(args.video, args.output, args.model, args.det_classes)
