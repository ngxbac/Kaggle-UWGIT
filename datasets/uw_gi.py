import numpy as np
import glob
import torch
import pandas as pd
from utils.misc import load_img, load_msk
import albumentations as A


def get_transform(dataset='train', image_sizes=[320, 384]):
    data_transforms = {
        "train": A.Compose([
            A.Resize(image_sizes[0], image_sizes[1]),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625,
                               scale_limit=0.05, rotate_limit=10, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),

            A.CoarseDropout(max_holes=8, max_height=image_sizes[0]//20, max_width=image_sizes[1]//20,
                            min_holes=5, fill_value=0, mask_fill_value=0, p=0.5),
        ], p=1.0),

        "valid": A.Compose([
            A.Resize(image_sizes[0], image_sizes[1]),
        ], p=1.0)
    }

    return data_transforms[dataset]


class UWGI(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv, fold=0, is_train=True, label=True, transforms=None):
        df = pd.read_csv(csv)
        if is_train:
            df = df[df.fold != fold]
        else:
            df = df[df.fold == fold]

        case_ids = df['case'].unique()
        self.images = []
        for case_id in case_ids:
            images = glob.glob(f"{data_dir}/{case_id}/*/*_image.npy")
            self.images += images
        self.transforms = transforms

    def __len__(self):
        return len(self.images)


    def get_mask(self, image):
        mask = image.replace('_image', '_mask')
        mask = np.load(mask)
        mask[mask !=0] = 1 # 3 x h x w
        # 0: large bowel
        # 1: small bowel
        # 2: stomach

        # Override large bowel to small bowel
        # large_bowel = mask[0] # h x w
        # large_bowel_idx = large_bowel == 1

        # small_bowel = mask[1]  # h x w
        # small_bowel[large_bowel_idx] = 0

        # mask[1, ...] = small_bowel

        mask = np.transpose(mask, (1, 2, 0))
        return mask

    def get_case_id(self, image_path):
        return image_path.split("/")[-3]

    def get_day(self, image_path):
        return image_path.split("/")[-2]

    def get_slice(self, image_path):
        return image_path.split("/")[-1]

    def __getitem__(self, index):
        image_path = self.images[index]
        mask = self.get_mask(image_path)
        image = np.load(image_path)
        image = image / image.max()

        ret = self.transforms(image=image, mask=mask)
        image = ret['image']
        mask = ret['mask']

        mask = mask.astype(np.float32)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)

        case_id = self.get_case_id(image_path)
        day = self.get_day(image_path)
        slice = self.get_slice(image_path)

        return {
            'image': image,
            'target': mask,
            'case_id': case_id,
            'day': day,
            'slice': slice
        }
