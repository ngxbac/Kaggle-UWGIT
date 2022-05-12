import pdb
import os
import numpy as np
import glob
import cv2
import pandas as pd
import SimpleITK


def get_spacing(path):
    filename = path.split("/")[-1][:-4]
    spacing_x = float(filename.split("_")[-2])
    spacing_y = float(filename.split("_")[-1])
    return (spacing_x, spacing_y, 1.5)


def get_hw(path):
    filename = path.split("/")[-1][:-4]
    h = int(filename.split("_")[-4])
    w = int(filename.split("_")[-3])
    return h, w


def get_id(path):
    case_day = path.split("/")[-3]
    filename = path.split("/")[-1][:-4]
    s = filename.split("_")[1]
    return f"{case_day}_slice_{s}"


def rle_decode(rle, mask, fill=255):
    s = rle.split()
    start, length = [np.asarray(x, dtype=int)
                     for x in (s[0:][::2], s[1:][::2])]
    start -= 1
    for i, l in zip(start, length):
        mask[i:i+l] = fill
    # mask = mask.reshape(width, height).T
    # mask = np.ascontiguousarray(mask)
    return mask


def merge_multi(mask_multi, mask):
    label_idx = np.where(mask != 0)[0]

    not_assigned_idx = label_idx[np.where(mask_multi[label_idx] == 0)[0]]
    assigned_idx = label_idx[np.where(mask_multi[label_idx] != 0)[0]]
    mask_multi[not_assigned_idx] = mask[not_assigned_idx]
    mask_multi[assigned_idx] += mask[assigned_idx] + 1

    return mask_multi


data_dir = "data/uw-madison-gi-tract-image-segmentation/train/"
df = pd.read_csv(f"{data_dir}/../train.csv")

save_dir = "data/nii-data-2/mask-multi/"
all_cases = os.listdir(data_dir)
for case in all_cases:
    case_dir = f"{data_dir}/{case}"
    all_days = os.listdir(case_dir)
    for day in all_days:
        day_dir = f"{case_dir}/{day}"
        all_frames = glob.glob(f"{day_dir}/scans/*.png")
        all_frames = sorted(all_frames)
        frames = []
        for frame in all_frames:
            h, w = get_hw(frame)
            spacing = get_spacing(frame)
            id = get_id(frame)
            tmp_df = df[df['id'] == id]
            segmentations = tmp_df['segmentation'].values
            class_ids = tmp_df['class'].values

            if h == 360 and w == 310:
                h = 310
                w = 360

            mask_array_multi = np.zeros(h*w, dtype=np.uint8)

            for class_id, segmentation in zip(class_ids, segmentations):
                mask_array = np.zeros(h*w, dtype=np.uint8)
                if class_id == 'stomach':
                    label = 1
                elif class_id == 'small_bowel':
                    label = 2
                elif class_id == 'large_bowel':
                    label = 3

                if segmentation is not np.nan:
                    mask_array = rle_decode(
                        rle=segmentation, mask=mask_array, fill=label)

                mask_array_multi = merge_multi(mask_array_multi, mask_array)

            mask_array_multi = mask_array_multi.reshape(h, w)
            mask_array_multi = np.ascontiguousarray(mask_array_multi)
            frames.append(mask_array_multi)

        frames = np.array(frames)
        print(len(frames), np.unique(frames), h, w, spacing)
        frames = SimpleITK.GetImageFromArray(frames)

        frames.SetOrigin((0.0, 0.0, 0.0))
        frames.SetSpacing(spacing)

        save_case_dir = f"{save_dir}/{case}/"
        os.makedirs(save_case_dir, exist_ok=True)

        save_case = f"{save_case_dir}/{case}_{day}.nii.gz"

        SimpleITK.WriteImage(
            frames, save_case
        )

        # save_case_dir = f"{save_dir}/{case}/"
        # os.makedirs(save_case_dir, exist_ok=True)

        # save_case = f"{save_case_dir}/{case}_{case}_{day}.nii.gz"

        # SimpleITK.WriteImage(
        #     frames, save_case
        # )
