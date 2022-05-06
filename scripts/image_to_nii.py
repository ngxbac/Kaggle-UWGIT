import SimpleITK
import cv2
import glob
import os
import numpy as np


data_dir = "data/uw-madison-gi-tract-image-segmentation/train/"
save_dir = "data/nii-data/train/"
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
            frame = cv2.imread(frame, cv2.IMREAD_UNCHANGED)
            frames.append(frame)

        frames = SimpleITK.GetImageFromArray(frames)
        # direction = np.array(
        # [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        # frames.SetDirection(direction)

        frames.SetOrigin((0.0, 0.0, 0.0))
        frames.SetSpacing((1.5, 1.5, 1.5))

        save_case_dir = f"{save_dir}/{case}/"
        os.makedirs(save_case_dir, exist_ok=True)

        save_case = f"{save_case_dir}/{case}_{day}.nii.gz"

        SimpleITK.WriteImage(
            frames, save_case
        )
