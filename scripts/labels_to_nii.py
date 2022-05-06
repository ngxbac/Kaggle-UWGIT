import pdb
import SimpleITK
import cv2
import glob
import random
import os
import numpy as np
import pandas as pd


data_dir = "data/uw-madison-gi-tract-image-segmentation/"
src = f'{data_dir}/train'
save_dir = "data/nii-data/mask/"

train = pd.read_csv(f'{data_dir}/train.csv')

not_nan_index = range(len(train))

# Let's see how many indice we got
print('The number of index is:', len(not_nan_index))

# There are only 3 classes
all_class = ['stomach', 'small_bowel', 'large_bowel']

# The empty list below will be used to store all the indice of each class
stomach = []
small_bowel = []
large_bowel = []

for i in range(len(not_nan_index)):
    row = not_nan_index[i]
    if train.loc[row, 'class'] == all_class[0]:
        stomach.append(not_nan_index[i])
    elif train.loc[row, 'class'] == all_class[1]:
        small_bowel.append(not_nan_index[i])
    else:
        large_bowel.append(not_nan_index[i])

# get all ids of each class
stomach_ids = []
small_bowel_ids = []
large_bowel_ids = []

for i in range(len(stomach)):
    stomach_ids.append(train.loc[stomach[i], 'id'])

for i in range(len(small_bowel)):
    small_bowel_ids.append(train.loc[small_bowel[i], 'id'])

for i in range(len(large_bowel)):
    large_bowel_ids.append(train.loc[large_bowel[i], 'id'])

# demonstrate some ids
print(stomach_ids[0:5])

# This dictionary stores the folders'name in each case. (e.g. case101_day20)
case_day = {}

all_case = os.listdir(src)
for i in range(len(all_case)):
    path = src + '/' + all_case[i]
    day = os.listdir(path)
    case_day[all_case[i]] = day

# This function is used to generate the filename of each image, and it also outputs the size of each image


def filename_generator(ID):
    split_id = ID.split('_')
    case = split_id[0]
    day_list = case_day[case]
    day = []
    for d in range(len(day_list)):
        day.append(day_list[d].split('_')[1])
    for k in range(len(day)):
        if split_id[1] == day[k]:
            dr = src + '/' + case + '/' + case + \
                '_' + split_id[1] + '/' + 'scans'
            h = os.listdir(dr)[0].split('_')[2]
            w = os.listdir(dr)[0].split('_')[3]
            break
    filename = src + '/' + split_id[0] + '/' + split_id[0] + '_' + split_id[1] + '/' + \
        'scans' + '/' + split_id[2] + '_' + split_id[3] + \
        '_' + h + '_' + w + '_1.50_1.50.png'
    size = [h, w]
    return filename, size


# These empty list are used to store something useful, according to the name of each list
stomach_fname = []
stomach_size = []
small_bowel_fname = []
small_bowel_size = []
large_bowel_fname = []
large_bowel_size = []
stomach_mask = []
small_bowel_mask = []
large_bowel_mask = []

# You can define some helper functions as a substitution for these for loops, but I think that didn't make much difference
for i in range(len(stomach_ids)):
    ID = stomach_ids[i]
    filename = filename_generator(ID)
    stomach_fname.append(filename[0])
    stomach_size.append(filename[1])

for i in range(len(small_bowel_ids)):
    ID = small_bowel_ids[i]
    filename = filename_generator(ID)
    small_bowel_fname.append(filename[0])
    small_bowel_size.append(filename[1])

for i in range(len(large_bowel_ids)):
    ID = large_bowel_ids[i]
    filename = filename_generator(ID)
    large_bowel_fname.append(filename[0])
    large_bowel_size.append(filename[1])

for i in range(len(stomach)):
    rle = train.loc[stomach[i], 'segmentation']
    stomach_mask.append(rle)

for i in range(len(small_bowel)):
    rle = train.loc[small_bowel[i], 'segmentation']
    small_bowel_mask.append(rle)

for i in range(len(large_bowel)):
    rle = train.loc[large_bowel[i], 'segmentation']
    large_bowel_mask.append(rle)


# We will use a Dataframe to store all the imformation we need.(e.g. height, width, maks, etc.)

# These lists store the information we need
stomach_info = []
small_bowel_info = []
large_bowel_info = []

for i in range(len(stomach)):
    ID = stomach_ids[i]
    name = 'stomach'
    path = stomach_fname[i]
    mask = stomach_mask[i]
    size = stomach_size[i]
    stomach_info.append([name, ID, path, mask, size])

for i in range(len(small_bowel)):
    ID = small_bowel_ids[i]
    name = 'small_bowel'
    path = small_bowel_fname[i]
    mask = small_bowel_mask[i]
    size = small_bowel_size[i]
    small_bowel_info.append([name, ID, path, mask, size])

for i in range(len(large_bowel)):
    ID = large_bowel_ids[i]
    name = 'large_bowel'
    path = large_bowel_fname[i]
    mask = large_bowel_mask[i]
    size = large_bowel_size[i]
    large_bowel_info.append([name, ID, path, mask, size])


# We need to concatenate these infos, then shuffle it.
info = stomach_info + small_bowel_info + large_bowel_info
random.shuffle(info)
print(len(info))

# Now we generate the Dataframe

classes = []
ids = []
path = []
mask = []
size = []

for i in range(len(info)):
    classes.append(info[i][0])
    ids.append(info[i][1])
    path.append(info[i][2])
    mask.append(info[i][3])
    size.append(info[i][4])

info_dict = {'class': classes,
             'id': ids,
             'path': path,
             'mask': mask,
             'size': size}
data = pd.DataFrame(info_dict,  columns=[
                    'class', 'id', 'path', 'mask', 'size'])
print(data.head())


# generate a Dataframe

info_dict = {'class': classes,
             'id': ids,
             'path': path,
             'mask': mask,
             'size': size}
data = pd.DataFrame(info_dict,  columns=[
                    'class', 'id', 'path', 'mask', 'size'])

data.to_csv('data.csv', index=False)

data['case'] = data['id'].apply(lambda x: x.split("_")[0])
data['slice'] = data['id'].apply(lambda x: x.split("_")[-1])
data['day'] = data['id'].apply(lambda x: x.split("_")[1])


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


# all_cases = os.listdir(data_dir)
# pdb.set_trace()
all_cases = data['case'].unique()
for case in all_cases:
    case_df = data[data['case'] == case]
    all_days = case_df['day'].unique()
    for day in all_days:
        day_df = case_df[case_df['day'] == day]
        day_df = day_df.sort_values(by=['slice'])
        all_masks = day_df['mask'].values
        all_sizes = day_df['size'].values
        all_classes = day_df['class'].values

        masks = []

        assert len(all_masks) % 3 == 0

        count = 0
        for mask, size, class_id in zip(all_masks, all_sizes, all_classes):
            h = int(size[0])
            w = int(size[1])

            if h == 360 and w == 310:
                h = 310
                w = 360
                print("aaa")

            if count == 0:
                mask_array = np.zeros(h*w, dtype=np.uint8)

            if class_id == 'stomach':
                label = 1
            elif class_id == 'small_bowel':
                label = 2
            elif class_id == 'large_bowel':
                label = 3

            # if not np.isnan(mask):
            # import pdb
            # pdb.set_trace()
            if mask is not np.nan:
                mask_array = rle_decode(rle=mask, mask=mask_array, fill=label)

            count += 1
            if count == 3:
                mask_array = mask_array.reshape(h, w)
                mask_array = np.ascontiguousarray(mask_array)
                masks.append(mask_array)
                count = 0

        print(len(masks), h, w)
        masks = SimpleITK.GetImageFromArray(masks)

        masks.SetOrigin((0.0, 0.0, 0.0))
        masks.SetSpacing((1.5, 1.5, 1.5))

        save_case_dir = f"{save_dir}/{case}/"
        os.makedirs(save_case_dir, exist_ok=True)

        save_case = f"{save_case_dir}/{case}_{case}_{day}.nii.gz"

        SimpleITK.WriteImage(
            masks, save_case
        )
