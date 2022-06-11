import numpy as np
import glob
import torch
import pandas as pd
from utils.misc import load_img, load_msk
import albumentations as A


def get_transform(dataset='train', image_sizes=[320, 384]):
    data_transforms = {
        "train": A.Compose([
            # A.Resize(image_sizes[0], image_sizes[1]),
            A.RandomResizedCrop(
                height=image_sizes[0],
                width=image_sizes[1],
                scale=(0.8, 1.0),
                ratio=(0.75, 1.3333333333333333),
                interpolation=1, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625,
                               scale_limit=0.05, rotate_limit=30, p=0.5),
            A.OneOf([
                A.GridDistortion(num_steps=5, distort_limit=0.05, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0)
            ], p=0.25),

            A.OneOf([
                A.MotionBlur(p=.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.25),

            # A.OneOf([
            #     A.CLAHE(clip_limit=2),
            #     A.IAASharpen(),
            #     A.IAAEmboss(),
            #     A.RandomBrightnessContrast(),
            # ], p=0.25),

            A.Cutout(num_holes=8, max_h_size=32,
                     max_w_size=32, fill_value=0, p=0.25),
            # A.Normalize()
        ], p=1.0),

        "valid": A.Compose([
            A.Resize(image_sizes[0], image_sizes[1]),
            # A.Normalize()
        ], p=1.0)
    }

    return data_transforms[dataset]


def cut_edge(data, threshold=0.1, percent=0.5):
    '''
    function that cuts zero edge
    '''
    H, W = data.shape
    H_s, H_e = 0, H - 1
    W_s, W_e = 0, W - 1

    while H_s < H:
        if data[H_s, :].max() > threshold:
            break
        H_s += 1
    while H_e > H_s:
        if data[H_e, :].max() > threshold:
            break
        H_e -= 1
    while W_s < W:
        if data[:, W_s].max() > threshold:
            break
        W_s += 1
    while W_e > W_s:
        if data[:, W_e].max() > threshold:
            break
        W_e -= 1

    delta_h = H_e - H_s
    delta_w = W_e - W_s

    if delta_h / H < percent or delta_w / W < percent:
        return 0, H, 0, W

    return int(H_s), int(H_e), int(W_s), int(W_e)


class UWGI(torch.utils.data.Dataset):
    def __init__(self, data_dir, csv, fold=0, multilabel=True, is_train=True, label=True, transforms=None, infer_pseudo=False):
        df = pd.read_csv(csv)

        if infer_pseudo:
            print("Infer pseudo!")
            df = df[df['is_pseudo'] == True]
        elif is_train:
            print("Training!")
            df = df[df.fold != fold]
        else:
            print("Validation!")
            df = df[df.fold == fold]

        # case_ids = df['case'].unique()
        # self.images = []
        # for case_id in case_ids:
        #     images = glob.glob(f"{data_dir}/{case_id}/*/*_image.npy")
        #     self.images += images

        df['mask'] = df['mask'].apply(
            lambda x: f"{data_dir}/{x}".replace("_mask", "_image"))
        self.images = df['mask'].values
        if 'is_pseudo' in df.columns:
            self.is_pseudos = df['is_pseudo'].values
        else:
            self.is_pseudos = [False] * len(df)

        self.transforms = transforms
        self.multilabel = multilabel
        self.infer_pseudo = infer_pseudo
        self.is_train = is_train

    def __len__(self):
        return len(self.images)

    def get_mask_multilabel(self, image):
        mask = image.replace('_image', '_mask')
        mask = np.load(mask)
        mask[mask != 0] = 1  # 3 x h x w
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

    def get_mask_multiclass(self, image, is_pseudo=False):
        if is_pseudo and not self.infer_pseudo:
            image = image.replace('uw-gi-25d', 'uw-gi-25d-pseudo')

        mask = image.replace('_image', '_mask')
        mask = np.load(mask)
        if not is_pseudo or self.infer_pseudo:
            mask = mask.max(axis=0)  # h x w

        return mask

    def get_case_id(self, image_path):
        return image_path.split("/")[-3]

    def get_day(self, image_path):
        return image_path.split("/")[-2]

    def get_slice(self, image_path):
        return image_path.split("/")[-1]

    def crop_roi(self, image, mask):
        channel = image[..., 1]
        ymin, ymax, xmin, xmax = cut_edge(channel)
        image = image[ymin:ymax, xmin:xmax]
        mask = mask[ymin:ymax, xmin:xmax]
        return image, mask

    def clipping(self, image, low, up):
        image = (image - low) / (up - low)
        image[image < 0] = 0
        image[image > 1] = 1
        return image.astype(np.float32)

    def organ_normalize(self, image):
        # image: h x w x 3

        organ_intensities = [
            [3, 8556],
            [5, 8358],
            [4, 10727]
        ]
        n_channels = image.shape[-1]
        all_channels = []
        for i in range(n_channels):
            channel = image[:, :, i]
            for low, up in organ_intensities:
                low, up = organ_intensities[i]
                channel = self.clipping(channel, low, up)
                all_channels.append(channel)

        all_channels = np.stack(all_channels, axis=-1).astype(np.float32)
        return all_channels

    def __getitem__(self, index):
        image_path = self.images[index]
        is_pseudo = self.is_pseudos[index]
        if self.multilabel:
            mask = self.get_mask_multilabel(image_path)
        else:
            mask = self.get_mask_multiclass(image_path, is_pseudo)

        image = np.load(image_path)
        # image = self.organ_normalize(image)
        image_h, image_w = image.shape[:2]
        image = image / image.max()
        image = image.astype(np.float32)

        # if np.random.rand() < 0.5 and self.is_train:
        #     image, mask = self.crop_roi(image, mask)

        # image = image * 255
        # image = image.astype(np.uint8)

        ret = self.transforms(image=image, mask=mask)
        image = ret['image']
        mask = ret['mask']

        mask = mask.astype(np.float32)
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)

        is_empty = mask.sum() == 0
        is_empty = np.array([is_empty]).astype(np.float32)

        if self.multilabel:
            mask = np.transpose(mask, (2, 0, 1)).astype(np.float32)
        else:
            mask = mask.astype(np.int64)

        case_id = self.get_case_id(image_path)
        day = self.get_day(image_path)
        slice = self.get_slice(image_path)

        return {
            'image': image,
            'target': mask,
            'empty': is_empty,
            'case_id': case_id,
            'day': day,
            'h': image_h,
            'w': image_w,
            'slice': slice
        }
