import argparse
import cv2
from tracker.boost_track import BoostTrack
import utils
import os
from args import make_parser
from default_settings import GeneralSettings, get_detector_path_and_im_size, BoostTrackPlusPlusSettings, BoostTrackSettings
import torch
from external.yolov10 import YoloV10Detector
import random

def preprocess_image(image, input_size):
    h, w = input_size
    image = cv2.resize(image, (w, h))
    image = image.astype('float32') / 255.0
    return torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)

def generate_random_color():
    return random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)

def process_video(video_path, output_path, model_path, det_classes, args):
    GeneralSettings.values['dataset'] = args.dataset
    GeneralSettings.values['use_embedding'] = not args.no_reid
    GeneralSettings.values['use_ecc'] = not args.no_cmc
    GeneralSettings.values['test_dataset'] = args.test_dataset

    BoostTrackSettings.values['s_sim_corr'] = args.s_sim_corr

    BoostTrackPlusPlusSettings.values['use_rich_s'] = not args.btpp_arg_iou_boost
    BoostTrackPlusPlusSettings.values['use_sb'] = not args.btpp_arg_no_sb
    BoostTrackPlusPlusSettings.values['use_vt'] = not args.btpp_arg_no_vt
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

    # Dictionary to store colors for each track ID
    id_colors = {}

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

        if pred is None or len(pred) == 0:
            out.write(frame)
            continue

        targets = tracker.update(pred, frame_rgb_tensor, frame_rgb, tag=f"frame_{frame_count}")
        tlwhs, ids, confs = utils.filter_targets(targets, GeneralSettings.conf_thresh, GeneralSettings.min_box_area)
        for tlwh, track_id, conf in zip(tlwhs, ids, confs):
            x1, y1, w, h = map(int, tlwh)

            # Assign a color to the track ID if not already assigned
            if track_id not in id_colors:
                id_colors[track_id] = generate_random_color()
            color = id_colors[track_id]

            cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), color, 2)
            label = f"ID: {int(track_id)}"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.write(frame)
        print(f"Processed frame {frame_count}\r", end="")

    cap.release()
    out.release()
    print("\nMOT pipeline completed!")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    parser = make_parser()
    parser.add_argument("--video", type=str, default="/kaggle/input/mot17-videos/MOT17-02-FRCNN-raw.mp4", help="Path to input video")
    parser.add_argument("--output", type=str, default="/kaggle/working/MOT17-02-FRCNN-raw.mp4", help="Path to save output video")
    parser.add_argument("--model", type=str, default="/kaggle/input/yolov10x/other/default/1/yolov10x.pt", help="Path to YOLOv10x model")
    parser.add_argument("--det_classes", nargs="+", type=int, default=[0], help="Detection classes (default: [0])")
    parser.add_argument("--dataset", type=str, default="mot17")
    parser.add_argument("--result_folder", type=str, default="results/trackers/")
    parser.add_argument("--test_dataset", action="store_true")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--no_reid", action="store_true", help="mark if visual embedding should NOT be used")
    parser.add_argument("--no_cmc", action="store_true", help="mark if camera motion compensation should NOT be used")
    parser.add_argument("--s_sim_corr", action="store_true", help="mark if you want to use corrected version of shape similarity calculation function")
    parser.add_argument("--btpp_arg_iou_boost", action="store_true", help="BoostTrack++ arg. Mark if only IoU should be used for detection confidence boost.")
    parser.add_argument("--btpp_arg_no_sb", action="store_true", help="BoostTrack++ arg. Mark if soft detection confidence boost should NOT be used.")
    parser.add_argument("--btpp_arg_no_vt", action="store_true", help="BoostTrack++ arg. Mark if varying threshold should NOT be used for the detection confidence boost.")
    parser.add_argument("--no_post", action="store_true", help="do not run post-processing.")
    args = parser.parse_args()

    process_video(args.video, args.output, args.model, args.det_classes, args)
