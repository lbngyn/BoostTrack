import os
import shutil
import time
import cv2
import dataset
import utils
from args import make_parser
from default_settings import GeneralSettings, get_detector_path_and_im_size, BoostTrackPlusPlusSettings, BoostTrackSettings
from external.adaptors import detector
from tracker.GBI import GBInterpolation
from tracker.boost_track import BoostTrack

# Import YOLOv10 detector class (giả sử bạn đã tạo module này)
from external.yolov10 import YoloV10Detector  # Module detector mới của bạn
import matplotlib.pyplot as plt
import torch



def get_main_args():
    parser = make_parser()
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
    if args.dataset == "mot17":
        args.result_folder = os.path.join(args.result_folder, "MOT17-val")
    elif args.dataset == "mot20":
        args.result_folder = os.path.join(args.result_folder, "MOT20-val")

    if args.test_dataset:
        args.result_folder.replace("-val", "-test")
    return args

def visulize(img): 

    # Chuyển đổi tensor thành numpy array
    img_numpy = img.cpu().numpy()

    # Nếu ảnh là 4D tensor (batch dimension), lấy ra ảnh đầu tiên
    if img_numpy.ndim == 4:
        img_numpy = img_numpy[0]

    # Hiển thị ảnh
    plt.imshow(img_numpy.transpose(1, 2, 0))  # Chuyển đổi CxHxW -> HxWxC cho matplotlib
    plt.show()



def main():
    # Set dataset and detector
    args = get_main_args()
    GeneralSettings.values['dataset'] = args.dataset
    GeneralSettings.values['use_embedding'] = not args.no_reid
    GeneralSettings.values['use_ecc'] = not args.no_cmc
    GeneralSettings.values['test_dataset'] = args.test_dataset

    BoostTrackSettings.values['s_sim_corr'] = args.s_sim_corr

    BoostTrackPlusPlusSettings.values['use_rich_s'] = not args.btpp_arg_iou_boost
    BoostTrackPlusPlusSettings.values['use_sb'] = not args.btpp_arg_no_sb
    BoostTrackPlusPlusSettings.values['use_vt'] = not args.btpp_arg_no_vt

    detector_path, size = get_detector_path_and_im_size(args)
    detector_path = '/kaggle/input/yolov10x/other/default/1/yolov10x.pt'
    # Thay thế YOLOX bằng YOLOv10x
    det = YoloV10Detector(model_path=detector_path, img_size=size)
    
    loader = dataset.get_mot_loader(args.dataset, args.test_dataset, size=size)
    
    tracker = None
    results = {}
    frame_count = 0
    total_time = 0

    for (img_real, np_img), label, info, idx in loader:
        frame_id = info[2].item()
        video_name = info[4][0].split("/")[0]

        if "FRCNN" not in video_name and args.dataset == "mot17":
            continue

        tag = f"{video_name}:{frame_id}"
        if video_name not in results:
            results[video_name] = []

        # img = img.cuda()
        img_path = os.path.join('/kaggle/input/mot17-converted-coco/MOT17/train', info[4][0])
        print(img_path)
        img = cv2.imread(img_path)
        print(img.shape)
        # Initialize tracker on first frame of a new video
        print(f"Processing {video_name}:{frame_id}\r", end="")
        if frame_id == 1:
            print(f"Initializing tracker for {video_name}")
            print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
            if tracker is not None:
                tracker.dump_cache()
            tracker = BoostTrack(video_name=video_name)

        # Sử dụng YOLOv10x để dự đoán
        pred = det.predict(img)
        print("Predict Value:", pred) 
        print("Predict type:", type(pred)) 
        print(len(pred)) 
        pred = pred.torch.tensor()

        start_time = time.time()

        if pred is None:
            continue

        # Update tracker
        targets = tracker.update(pred, img_real, img, tag)
        tlwhs, ids, confs = utils.filter_targets(targets, GeneralSettings['aspect_ratio_thresh'], GeneralSettings['min_box_area'])

        total_time += time.time() - start_time
        frame_count += 1

        results[video_name].append((frame_id, tlwhs, ids, confs))

    print(f"Time spent: {total_time:.3f}, FPS {frame_count / (total_time + 1e-9):.2f}")
    print(total_time)

    # Save detector results
    det.dump_cache()
    tracker.dump_cache()

    folder = os.path.join(args.result_folder, args.exp_name, "data")
    os.makedirs(folder, exist_ok=True)
    for name, res in results.items():
        result_filename = os.path.join(folder, f"{name}.txt")
        utils.write_results_no_score(result_filename, res)

    print(f"Finished, results saved to {folder}")

    if not args.no_post:
        post_folder = os.path.join(args.result_folder, args.exp_name + "_post")
        pre_folder = os.path.join(args.result_folder, args.exp_name)
        if os.path.exists(post_folder):
            print(f"Overwriting previous results in {post_folder}")
            shutil.rmtree(post_folder)
        shutil.copytree(pre_folder, post_folder)
        post_folder_data = os.path.join(post_folder, "data")
        interval = 1000
        utils.dti(post_folder_data, post_folder_data, n_dti=interval, n_min=25)
        print(f"Linear interpolation post-processing applied, saved to {post_folder_data}.")

        res_folder = os.path.join(args.result_folder, args.exp_name, "data")
        post_folder_gbi = os.path.join(args.result_folder, args.exp_name + "_post_gbi", "data")

        if not os.path.exists(post_folder_gbi):
            os.makedirs(post_folder_gbi)
        for file_name in os.listdir(res_folder):
            in_path = os.path.join(post_folder_data, file_name)
            out_path2 = os.path.join(post_folder_gbi, file_name)

            GBInterpolation(
                path_in=in_path,
                path_out=out_path2,
                interval=interval
            )
        print(f"Gradient boosting interpolation post-processing applied, saved to {post_folder_gbi}.")


if __name__ == "__main__":
    main()
