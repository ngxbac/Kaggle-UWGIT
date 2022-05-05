import cv2
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import Dataset

data_root = "data/uw-madison-gi-tract-image-segmentation"


class GTTractDataset(Dataset):
    def __init__(self, transform=None, fold=0, is_train=True):
        kfold_df = pd.read_csv(f"{data_root}/kfold.csv")

        if is_train:
            kfold_df = kfold_df[kfold_df['fold'] != fold]
        else:
            kfold_df = kfold_df[kfold_df['fold'] == fold]

        cases = kfold_df.case.values

        # self.images = self.df['image'].values
        # self.labels = self.df['label'].values
        self.transform = transform

    def load_data(self):
        df = pd.read_csv(f'{data_root}/train.csv')
        df['case'] = df['id'].apply(lambda x: x.split("_")[0])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transform:
            image = self.transform(image)

        return {
            'image': image,
            'label': np.array([label]).astype(np.float32)
        }

