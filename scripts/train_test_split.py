import pandas as pd
import os
from glob import glob
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


n_fold = 5


path_df = pd.DataFrame(glob(
    'data/uwmgi-25d-stride2-dataset/images/images/*'), columns=['image_path'])
path_df['mask_path'] = path_df.image_path.str.replace('image', 'mask')
path_df['id'] = path_df.image_path.map(
    lambda x: x.split('/')[-1].replace('.npy', ''))
path_df.head()


df = pd.read_csv('data/uwmgi-mask-dataset/train.csv')
df['segmentation'] = df.segmentation.fillna('')
df['rle_len'] = df.segmentation.map(len)  # length of each rle mask

df2 = df.groupby(['id'])['segmentation'].agg(
    list).to_frame().reset_index()  # rle list of each id
df2 = df2.merge(df.groupby(['id'])['rle_len'].agg(
    sum).to_frame().reset_index())  # total length of all rles of each id

df = df.drop(columns=['segmentation', 'class', 'rle_len'])
df = df.groupby(['id']).head(1).reset_index(drop=True)
df = df.merge(df2, on=['id'])
df['empty'] = (df.rle_len == 0)  # empty masks

df = df.drop(columns=['image_path', 'mask_path'])
df = df.merge(path_df, on=['id'])
df.head()


fault1 = 'case7_day0'
fault2 = 'case81_day30'
df = df[~df['id'].str.contains(fault1) & ~df['id'].str.contains(
    fault2)].reset_index(drop=True)
df.head()


skf = StratifiedGroupKFold(
    n_splits=n_fold, shuffle=True, random_state=2411)
for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['empty'], groups=df["case"])):
    df.loc[val_idx, 'fold'] = fold

df.to_csv('train_valid.csv', index=False)
