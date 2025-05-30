"""
detect zoom in/zoom out videos
"""

import argparse

# cv2.setNumThreads(1)
from functools import partial

import cv2
import pandas as pd
import scipy.stats as stats
from pandarallel import pandarallel
from tqdm import tqdm

# thresholds
max_frames = 5  # max number of times to run correlation analysis
p_threshold = 0.1
tau_threshold = 0.50  # correlation above which to determine as zoom
crop_ratio = 2 / 3
zoom_frame_count_threshold = 0.4  # % of frames above which has zoom
image_size = (256, 256)
min_corr_counts = 10  # minimum number of points required to calc correlation
sample_interval = 10  # need to be larger than 10 to be accurate


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path", type=str, help="Path to the input CSV file")
    parser.add_argument("--num_workers", type=int, default=None, help="#workers for pandarallel")
    parser.add_argument("--disable_parallel", action="store_true", help="disable parallel processing")
    parser.add_argument("--num_samples", type=int, default=-1, help="number of samples to process, for quick tests")

    args = parser.parse_args()
    return args


def crop_resize(image, image_size):
    # crop the edge to remove watermarks
    h, w = image.shape
    crop_h, crop_w = int(h * crop_ratio), int(w * crop_ratio)
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    image = image[start_h : start_h + crop_h, start_w : start_w + crop_w]

    # resize
    h, w = image.shape
    h_t, w_t = image_size
    scale = max(h_t / h, w_t / w)
    resized_image = cv2.resize(image, (int(w * scale), int(h * scale)))  # NOTE: strage but cv2 need to put w h

    # center crop
    h, w = resized_image.shape[:2]
    # Calculate the center crop dimensions
    crop_h, crop_w = image_size
    start_h = (h - crop_h) // 2
    start_w = (w - crop_w) // 2
    center_cropped_image = resized_image[start_h : start_h + crop_h, start_w : start_w + crop_w]

    return center_cropped_image


def is_dynamic_video(
    video_path,
):
    cap = cv2.VideoCapture(video_path)

    # Check if the video opened successfully
    if not cap.isOpened():
        print(f"Error: Could not open video '{video_path}'")
        return False

    frame_index = -1

    p0 = None
    while p0 is None:
        frame_index += 1
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        if not success:
            print(f"Warning: Failed to read frame {frame_index} of video '{video_path}'")
            break

        prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = crop_resize(prev_gray, image_size)

        # Feature parameters
        feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **feature_params)
    if p0 is None:
        return True

    # Calculate flow
    # Parameters for optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    zoom_count = 0
    analyzed_count = 0
    invalid_count = 0

    while analyzed_count < max_frames:
        if invalid_count >= 3:
            break

        frame_index += sample_interval

        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        success, frame = cap.read()
        if not success:
            break

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = crop_resize(gray, image_size)

        # Calculate optical flow
        p1, status, err = cv2.calcOpticalFlowPyrLK(prev_gray, gray, p0, None, **lk_params)

        if p1 is None:
            invalid_count += 1
            continue

        # Select good points
        good_new = p1[status == 1]  # status == 1: found flow; status == 0: didn't
        good_old = p0[status == 1]

        # Calculate motion vectors
        motion_vectors = good_new - good_old

        motion_x_neg = []
        x_neg = []
        motion_x_pos = []
        x_pos = []
        motion_y_neg = []
        y_neg = []
        motion_y_pos = []
        y_pos = []

        for i, (motion_x, motion_y) in enumerate(motion_vectors):
            if motion_x < 0:
                motion_x_neg.append(motion_x)
                x_neg.append(good_old[i][0])
            else:
                motion_x_pos.append(motion_x)
                x_pos.append(good_old[i][0])

            if motion_y < 0:
                motion_y_neg.append(motion_y)
                y_neg.append(good_old[i][1])
            else:
                motion_y_pos.append(motion_y)
                y_pos.append(good_old[i][1])

        is_zoom = True
        corr_checks = 0

        if len(motion_x_neg) > min_corr_counts:
            corr_checks += 1
            tau_x_neg, p_x_neg = stats.kendalltau(motion_x_neg, x_neg)
            if p_x_neg > p_threshold or abs(tau_x_neg) < tau_threshold:
                is_zoom = False
        if len(motion_x_pos) > min_corr_counts:
            corr_checks += 1
            tau_x_pos, p_x_pos = stats.kendalltau(motion_x_pos, x_pos)
            if p_x_pos > p_threshold or abs(tau_x_pos) < tau_threshold:
                is_zoom = False
        if len(motion_y_neg) > min_corr_counts:
            corr_checks += 1
            tau_y_neg, p_y_neg = stats.kendalltau(motion_y_neg, y_neg)
            if p_y_neg > p_threshold or abs(tau_y_neg) < tau_threshold:
                is_zoom = False
        if len(motion_y_pos) > min_corr_counts:
            corr_checks += 1
            tau_y_pos, p_y_pos = stats.kendalltau(motion_y_pos, y_pos)
            if p_y_pos > p_threshold or abs(tau_y_pos) < tau_threshold:
                is_zoom = False

        if corr_checks > 0:
            if is_zoom:
                zoom_count += 1
            analyzed_count += 1
        else:
            invalid_count += 1

        # Update previous frame and points
        prev_gray = gray.copy()
        p0 = good_new.reshape(-1, 1, 2)

    cap.release()

    if analyzed_count > 2 and zoom_count / analyzed_count >= zoom_frame_count_threshold:
        return False
    else:
        return True


def process_single_row(row, args):
    path = row["path"]
    return is_dynamic_video(path)


def main():
    args = parse_args()
    meta_path = args.meta_path

    tqdm.pandas()
    if not args.disable_parallel:
        if args.num_workers is not None:
            pandarallel.initialize(progress_bar=True, nb_workers=args.num_workers)
        else:
            pandarallel.initialize(progress_bar=True)
    process_single_row_partial = partial(process_single_row, args=args)

    meta = pd.read_csv(meta_path)
    if args.num_samples > -1:
        meta = meta[: args.num_samples]

    if not args.disable_parallel:
        ret = meta.parallel_apply(process_single_row_partial, axis=1)
    else:
        ret = meta.progress_apply(process_single_row_partial, axis=1)
    meta['zoom'] = ret

    out_path = meta_path.replace(".csv", "_zoom.csv")
    meta.to_csv(out_path, index=False)
    print(f"New meta saved to '{out_path}'")

if __name__ == "__main__":
    main()
