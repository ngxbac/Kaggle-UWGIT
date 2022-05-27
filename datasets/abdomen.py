import numpy as np
import glob
import torch
import pandas as pd
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


class Abdomen(torch.utils.data.Dataset):
    def __init__(self, data_dir, transforms=None):
        self.images = glob.glob(f"{data_dir}/*/image.*.npy")
        self.transforms = transforms

    def __len__(self):
        return len(self.images)

    def get_mask_multiclass(self, image):
        mask = image.replace('image', 'mask')
        mask = np.load(mask)
        return mask


    def __getitem__(self, index):
        image_path = self.images[index]
        mask = self.get_mask_multiclass(image_path)

        image = np.load(image_path)
        # image_min = image.min()
        # image_max = image.max()
        # image = (image - image_min) / (image_max - image_min)
        image = image / image.max()
        image = np.stack((image, image, image), axis=-1).astype(np.float32)
        # print(image.shape, mask.shape, mask.min(), mask.max())

        ret = self.transforms(image=image, mask=mask)
        image = ret['image']
        mask = ret['mask']

        mask = mask.astype(np.float32)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        mask = mask.astype(np.int)

        return {
            'image': image,
            'target': mask,
        }
