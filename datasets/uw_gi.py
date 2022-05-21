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

        case_ids = df['case'].values
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
        mask = np.transpose(mask, (1, 2, 0))
        return mask

    def __getitem__(self, index):
        image = self.images[index]
        mask = self.get_mask(image)
        image = np.load(image)
        image = image / image.max()


        ret = self.transforms(image=image, mask=mask)
        image = ret['image']
        mask = ret['mask']

        mask = mask.astype(np.float32)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)
        # print(mask.shape, image.shape)

        return {
            'image': image,
            'target': mask
        }
