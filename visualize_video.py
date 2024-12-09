import cv2
import time
import os
from external.adaptors import detector
from tracker.boost_track import BoostTrack
import utils
from default_settings import GeneralSettings, get_detector_path_and_im_size

def process_video(input_video_path, output_video_path, detector_model="yolox"):
    # Initialize detector
    detector_path, size = ("external/weights/bytetrack_x_mot17.pth.tar", (800, 1440))  # Example usage
    det = detector.Detector(detector_model, detector_path, "mot17")

    # Initialize video capture and writer
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        print(f"Error: Cannot open video {input_video_path}")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for .mp4 files

    # Initialize video writer
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    print(f"Processing video: {input_video_path}")

    # Initialize tracker
    tracker = BoostTrack(video_name="custom_video")
    frame_count = 0
    total_time = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Object detection
        start_time = time.time()
        pred = det(frame_rgb, f"frame_{frame_count}")
        if pred is None:
            out.write(frame)
            continue

        # Update tracker
        targets = tracker.update(pred, frame_rgb, frame, f"frame_{frame_count}")
        tlwhs, ids, confs = utils.filter_targets(targets, GeneralSettings['aspect_ratio_thresh'], GeneralSettings['min_box_area'])

        total_time += time.time() - start_time

        # Draw results on frame
        for tlwh, track_id, conf in zip(tlwhs, ids, confs):
            x1, y1, w, h = map(int, tlwh)
            color = (0, 255, 0)  # Green for bounding boxes
            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
            label = f"ID: {track_id} ({conf:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # Write frame to output video
        out.write(frame)

        print(f"Processed frame {frame_count}\r", end="")

    # Release resources
    cap.release()
    out.release()
    print("\nProcessing complete!")
    print(f"Output saved to: {output_video_path}")
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {frame_count / (total_time + 1e-9):.2f}")

# Example usage
input_video = "/kaggle/input/mot17-videos/MOT17-01-FRCNN-raw.mp4"
output_video = "/kaggle/working/processed_videos/MOT17-01-FRCNN-raw.mp4"
process_video(input_video, output_video)
