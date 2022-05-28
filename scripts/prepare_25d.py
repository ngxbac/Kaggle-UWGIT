import pandas as pd
import numpy as np
import cv2
import os
import glob


def load_image_(data_dir, case_id, day, slice):
    image = glob.glob(f"{data_dir}/train/{case_id}/{case_id}_{day}/scans/{slice}*")
    if len(image) == 0:
        return None

    image = image[0]
    image = cv2.imread(image, cv2.IMREAD_UNCHANGED)
    return image


def load_image(data_dir, case_id, day, slice, stride):
    slice_number = int(slice.split("_")[1])
    cur_image = load_image_(data_dir, case_id, day, slice)
    assert cur_image is not None

    all_images = []
    for i in range(stride, 0, -1):
        prev_slice = slice_number - i
        prev_slice = str(prev_slice).zfill(4)
        prev_slice = f"slice_{prev_slice}"
        prev_image = load_image_(data_dir, case_id, day, prev_slice)
        prev_image = cur_image if prev_image is None else prev_image
        all_images.append(prev_image)

    all_images.append(cur_image)

    for i in range(1, stride + 1):
        next_slice = slice_number + i
        next_slice = str(next_slice).zfill(4)
        next_slice = f"slice_{next_slice}"
        next_image = load_image_(data_dir, case_id, day, next_slice)
        next_image = cur_image if next_image is None else next_image
        all_images.append(next_image)

    image = np.array(all_images) # 3 x h x w
    image = np.transpose(image, (1, 2, 0)) # h x w x 3
    return image


def rle_decode(rle, mask, fill=255):
    s = rle.split()
    start, length = [np.asarray(x, dtype=int)
                     for x in (s[0:][::2], s[1:][::2])]
    start -= 1
    for i, l in zip(start, length):
        mask[i:i+l] = fill
    return mask


stride = 2

data_dir = "data/uw-madison-gi-tract-image-segmentation"
save_data_dir = "data/uw-gi-25d-5"

df = pd.read_csv(f"{data_dir}/train.csv")
ids = df['id'].values
classes = df['class'].values
segmentation = df['segmentation'].values

n_images = len(ids)

for i in range(n_images):
    id, cls, seg = ids[i], classes[i], segmentation[i]
    case_id = id.split("_")[0]
    day = id.split("_")[1]
    slice = id.replace(f"{case_id}_", "").replace(f"{day}_", "")


    if i % 3 == 0:
        image = load_image(data_dir, case_id, day, slice, stride)
        mask_array = []

    h, w = image.shape[:2]
    mask = np.zeros(h*w, dtype=np.uint8)
    if cls == 'stomach':
        label = 3
    elif cls == 'small_bowel':
        label = 2
    elif cls == 'large_bowel':
        label = 1

    if seg is not np.nan:
        mask = rle_decode(
            rle=seg,
            mask=mask,
            fill=label
        )

    mask = mask.reshape(h, w)
    mask = np.ascontiguousarray(mask)
    mask_array.append(mask)

    if i % 3 == 2:
        assert len(mask_array) == 3
        mask_array = np.array(mask_array)

        save_folder = f"{save_data_dir}/{case_id}/{day}/"
        os.makedirs(save_folder, exist_ok=True)
        np.save(f"{save_folder}/{slice}_image.npy", image)
        np.save(f"{save_folder}/{slice}_mask.npy", mask_array)
