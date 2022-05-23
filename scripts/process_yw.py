import sys

# sys.path.append('../input/monai-v081/')

import pandas as pd
import json
import torch
import os
import numpy as np
from glob import glob

# Open the training dataframe and display the initial dataframe
# the way to process the df refers to:
# https://www.kaggle.com/code/dschettler8845/uwmgit-deeplabv3-end-to-end-pipeline-tf

DATA_DIR = "./data/uw-madison-gi-tract-image-segmentation/"

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
train_df = pd.read_csv(TRAIN_CSV)

# Get all training images
all_train_images = glob(os.path.join(DATA_DIR, "train", "**", "*.png"), recursive=True)


def get_filepath_from_partial_identifier(_ident, file_list):
    return [x for x in file_list if _ident in x][0]

def df_preprocessing(df, globbed_file_list, is_test=False):
    """ The preprocessing steps applied to get column information """
    # 1. Get Case-ID as a column (str and int)
    df["case_id_str"] = df["id"].apply(lambda x: x.split("_", 2)[0])

    # 2. Get Day as a column
    df["day_num_str"] = df["id"].apply(lambda x: x.split("_", 2)[1])

    # 3. Get Slice Identifier as a column
    df["slice_id"] = df["id"].apply(lambda x: x.split("_", 2)[2])

    # 4. Get full file paths for the representative scans
    df["_partial_ident"] = (globbed_file_list[0].rsplit("/", 4)[0]+"/"+ # /kaggle/input/uw-madison-gi-tract-image-segmentation/train/
                           df["case_id_str"]+"/"+ # .../case###/
                           df["case_id_str"]+"_"+df["day_num_str"]+ # .../case###_day##/
                           "/scans/"+df["slice_id"]) # .../slice_####
    _tmp_merge_df = pd.DataFrame({"_partial_ident":[x.rsplit("_",4)[0] for x in globbed_file_list], "f_path":globbed_file_list})
    df = df.merge(_tmp_merge_df, on="_partial_ident").drop(columns=["_partial_ident"])

    # 5. Get slice dimensions from filepath (int in pixels)
    df["slice_w"] = df["f_path"].apply(lambda x: int(x[:-4].rsplit("_",4)[2]))
    df["slice_h"] = df["f_path"].apply(lambda x: int(x[:-4].rsplit("_",4)[1]))

    # 6. Pixel spacing from filepath (float in mm)
    df["px_spacing_h"] = df["f_path"].apply(lambda x: float(x[:-4].rsplit("_",4)[3]))
    df["px_spacing_w"] = df["f_path"].apply(lambda x: float(x[:-4].rsplit("_",4)[4]))

    if not is_test:
        # 7. Merge 3 Rows Into A Single Row (As This/Segmentation-RLE Is The Only Unique Information Across Those Rows)
        l_bowel_df = df[df["class"]=="large_bowel"][["id", "segmentation"]].rename(columns={"segmentation":"lb_seg_rle"})
        s_bowel_df = df[df["class"]=="small_bowel"][["id", "segmentation"]].rename(columns={"segmentation":"sb_seg_rle"})
        stomach_df = df[df["class"]=="stomach"][["id", "segmentation"]].rename(columns={"segmentation":"st_seg_rle"})
        df = df.merge(l_bowel_df, on="id", how="left")
        df = df.merge(s_bowel_df, on="id", how="left")
        df = df.merge(stomach_df, on="id", how="left")
        df = df.drop_duplicates(subset=["id",]).reset_index(drop=True)

    # 8. Reorder columns to the a new ordering (drops class and segmentation as no longer necessary)
    new_col_order = ["id", "f_path",
                     "lb_seg_rle",
                     "sb_seg_rle",
                     "st_seg_rle",
                     "slice_h", "slice_w", "px_spacing_h",
                     "px_spacing_w", "case_id_str",
                     "day_num_str", "slice_id",]
    if is_test: new_col_order.insert(1, "class")
    new_col_order = [_c for _c in new_col_order if _c in df.columns]
    df = df[new_col_order]

    return df


train_df = df_preprocessing(train_df, all_train_images)
# I use the same data splits as AWSAF does
# the splits.csv refers to:
# https://www.kaggle.com/code/awsaf49/uwmgi-unet-train-pytorch/
splits = pd.read_csv("data/split.csv")

train_df = train_df.merge(splits, on="id")
train_df["fold"] = train_df["fold"].astype(np.uint8)


# ref: https://www.kaggle.com/paulorzp/run-length-encode-and-decode
# modified from: https://www.kaggle.com/inversion/run-length-decoding-quick-start
def rle_decode(mask_rle, shape, color=1):
    """ TBD

    Args:
        mask_rle (str): run-length as string formated (start length)
        shape (tuple of ints): (height,width) of array to return

    Returns:
        Mask (np.array)
            - 1 indicating mask
            - 0 indicating background

    """
    # Split the string by space, then convert it into a integer array
    s = np.array(mask_rle.split(), dtype=int)

    # Every even value is the start, every odd value is the "run" length
    starts = s[0::2] - 1
    lengths = s[1::2]
    ends = starts + lengths

    # The image image is actually flattened since RLE is a 1D "run"
    if len(shape)==3:
        h, w, d = shape
        img = np.zeros((h * w, d), dtype=np.float32)
    else:
        h, w = shape
        img = np.zeros((h * w,), dtype=np.float32)

    # The color here is actually just any integer you want!
    for lo, hi in zip(starts, ends):
        img[lo : hi] = color

    # Don't forget to change the image back to the original shape
    return img.reshape(shape)


def load_img_mask(l):
    img_data = loader(l.f_path)
    img_h, img_w = img_data[0].shape
    shape = (l.slice_h, l.slice_w)
    assert shape == (img_h, img_w)
    wh_shape = (img_w, img_h)
    if pd.isna(l.lb_seg_rle):
        lb_mask = np.zeros(wh_shape)
    else:
        lb_mask = rle_decode(l.lb_seg_rle, wh_shape)

    if pd.isna(l.sb_seg_rle):
        sb_mask = np.zeros(wh_shape)
    else:
        sb_mask = rle_decode(l.sb_seg_rle, wh_shape)

    if pd.isna(l.st_seg_rle):
        st_mask = np.zeros(wh_shape)
    else:
        st_mask = rle_decode(l.st_seg_rle, wh_shape)

    all_mask = np.stack([lb_mask, sb_mask, st_mask], axis=0).astype(np.uint8)
    # multiclass mask,
    mask_arr = st_mask*3
    mask_arr = np.where(sb_mask==1, 2, mask_arr)
    mask_arr = np.where(lb_mask==1, 1, mask_arr)

    return img_data[0], all_mask, mask_arr


from monai.transforms import LoadImage
from monai.data import NibabelWriter

loader = LoadImage()

output_dir = "data/ywdata/"
os.makedirs(output_dir, exist_ok=True)

data_3d_info = []
ct = 0
for group in train_df.groupby(["case_id_str", "day_num_str"]):

    case_3d_img, case_3d_mask, case_3d_mask_multiclass = [], [], []

    case_id_str, day_num_str = group[0]
    group_id = case_id_str + "_" + day_num_str
    group_df = group[1].sort_values("slice_id", ascending=True)
    n_slices = group_df.shape[0]
    for idx in range(n_slices):
        slc = group_df.iloc[idx]
        slc_img, slc_mask, slc_multiclass_mask = load_img_mask(slc)
        case_3d_img.append(slc_img)
        case_3d_mask.append(slc_mask)
        case_3d_mask_multiclass.append(slc_multiclass_mask)

    case_3d_img = np.stack(case_3d_img, axis=-1)
    case_3d_mask = np.stack(case_3d_mask, axis=-1)
    case_3d_mask = np.transpose(case_3d_mask, [2, 1, 3, 0]) # c w h d to h w d c
    case_3d_mask_multiclass = np.stack(case_3d_mask_multiclass, axis=-1)
    case_3d_mask_multiclass = np.transpose(case_3d_mask_multiclass, [1, 0, 2]) # w h d to h w d

    assert np.all(case_3d_mask.astype(np.uint8) == case_3d_mask)
    case_3d_mask = case_3d_mask.astype(np.uint8)

    if case_3d_mask.shape[:-1] != case_3d_img.shape:
        print("shape not match on group: ", group_id)

    group_spacing = group[1][["px_spacing_h"]].values[0][0]

    group_affine = np.eye(4) * group_spacing
    group_affine[-1][-1] = 1.5  # all z-axis spacing is 1.5
    group_fold = group[1][["fold"]].values[0][0]

    group_root_dir = os.path.join(output_dir, "train", case_id_str, group_id)
    os.makedirs(group_root_dir)
    # write image
    writer = NibabelWriter()
    writer.set_data_array(case_3d_img, channel_dim=None)
    writer.set_metadata({"affine": group_affine, "original_affine": group_affine, "dtype": np.int16})
    writer.write(f"{group_root_dir}/{group_id}_image.nii.gz", verbose=False)

    # write mask
    writer = NibabelWriter()
    writer.set_data_array(case_3d_mask, channel_dim=-1)
    writer.set_metadata({"affine": group_affine, "original_affine": group_affine, "dtype": np.uint8})
    writer.write(f"{group_root_dir}/{group_id}_mask.nii.gz", verbose=False)

    # write mask multiclass
    writer = NibabelWriter()
    writer.set_data_array(case_3d_mask_multiclass, channel_dim=None)
    writer.set_metadata({"affine": group_affine, "original_affine": group_affine, "dtype": np.uint8})
    writer.write(f"{group_root_dir}/{group_id}_mask_multiclass.nii.gz", verbose=False)

    data_3d_info.append({
        "id": group_id,
        "fold": group_fold,
        "image_path": f"{group_root_dir}/{group_id}_image.nii.gz",
        "mask_path": f"{group_root_dir}/{group_id}_mask.nii.gz",
        "mask_multiclass_path": f"{group_root_dir}/{group_id}_mask_multiclass.nii.gz",
    })

    ct += 1
    print("finish: ", ct, " shape: ", case_3d_mask.shape)

data_3d_info = pd.DataFrame(data_3d_info)
data_3d_info.to_csv("data/ywdata/data_3d_info.csv", index=False)


for fold in range(5):
    train_data, val_data = [], []
    train_df = data_3d_info[data_3d_info["fold"] != fold]
    val_df = data_3d_info[data_3d_info["fold"] == fold]

    for line in train_df.values:
        train_data.append({"image": line[2], "mask": line[3], "mask_multiclass": line[4], "id": line[0]})
    for line in val_df.values:
        val_data.append({"image": line[2], "mask": line[3], "mask_multiclass": line[4], "id": line[0]})

    all_data = {"train": train_data, "val": val_data}

    with open(f"data/ywdata/dataset_3d_fold_{fold}.json", 'w') as f:
        json.dump(all_data, f)
