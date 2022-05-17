# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import math
import numpy as np
import torch
import pandas as pd
from monai import transforms, data
# from monai.data import load_decathlon_datalist


class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None,
                 shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError(
                    "Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(
            math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(
            indices[self.rank:self.total_size:self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[:(self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(
                        indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch


class ConvertToMultiChannel(transforms.MapTransform):
    """
    Convert labels to multi channels based on brats classes:
    label 1 is the peritumoral edema
    label 2 is the GD-enhancing tumor
    label 3 is the necrotic and non-enhancing tumor core
    The possible classes are TC (Tumor core), WT (Whole tumor)
    and ET (Enhancing tumor).

    """

    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = [None] * 3

            for i in range(1, 4):
                result[i-1] = d[key] == i

            for i in range(4, 7):
                idx = d[key] == i
                if i == 4:  # 1 + 2 + 1
                    result[0][idx] = 1
                    result[1][idx] = 1

                if i == 5:  # 1 + 3 + 1
                    result[0][idx] = 1
                    result[2][idx] = 1

                if i == 6:  # 2 + 3 + 1
                    result[1][idx] = 1
                    result[2][idx] = 1

            result = np.concatenate(result, axis=0)
            d[key] = result
        return d


def get_loader(args):
    from glob import glob
    from sklearn.model_selection import KFold
    data_dir = args.data_dir
    print(data_dir)

    train_fold = args.fold
    df = pd.read_csv('train_valid_case.csv')
    train_df = df[df['fold'] != train_fold]
    valid_df = df[df['fold'] == train_fold]

    train_cases = train_df['case'].unique()
    valid_cases = valid_df['case'].unique()

    images = sorted(glob(os.path.join(data_dir, "train/*/*.nii.gz")))

    # all_cases = [image.split("_")[-2] for image in images]
    # unique_cases = np.unique(all_cases)
    # kf = KFold(n_splits=5, random_state=2411, shuffle=True)
    # for fold, (train_idx, valid_idx) in enumerate(kf.split(unique_cases)):
    #     if fold == train_fold:
    #         train_cases = unique_cases[train_idx]
    #         valid_cases = unique_cases[valid_idx]

    if args.multilabel:
        mask_folder = 'mask-multi'
    else:
        mask_folder = 'mask'

    train_images = [image for image in images if image.split(
        "/")[-2] in train_cases]
    train_segs = [image.replace('train', mask_folder)
                  for image in train_images]

    val_images = [image for image in images if image.split(
        "/")[-2] in valid_cases]
    val_segs = [image.replace('train', mask_folder) for image in val_images]

    train_files = [{"image": img, "label": seg}
                   for img, seg in zip(train_images, train_segs)]
    val_files = [{"image": img, "label": seg}
                 for img, seg in zip(val_images, val_segs)]

    base_transforms = [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.AddChanneld(keys=["image", "label"]),
            transforms.Orientationd(keys=["image", "label"],
                                    axcodes="RAS"),
            transforms.ScaleIntensityRanged(
                keys=["image"], a_min=0, a_max=16384, b_min=0.0, b_max=1.0, clip=True),
            transforms.CropForegroundd(
                keys=["image", "label"], source_key="image"),
    ]

    advanced_transforms = [
            transforms.RandCropByPosNegLabeld(
                keys=["image", "label"],
                label_key="label",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
            ),

            # transforms.RandFlipd(keys=["image", "label"],
            #                      prob=args.RandFlipd_prob,
            #                      spatial_axis=0),
            # transforms.RandFlipd(keys=["image", "label"],
            #                      prob=args.RandFlipd_prob,
            #                      spatial_axis=1),
            # transforms.RandFlipd(keys=["image", "label"],
            #                      prob=args.RandFlipd_prob,
            #                      spatial_axis=2),
            # transforms.RandRotate90d(
            #     keys=["image", "label"],
            #     prob=args.RandRotate90d_prob,
            #     max_k=3,
            # ),
            # transforms.RandScaleIntensityd(keys="image",
            #                                factors=0.1,
            #                                prob=args.RandScaleIntensityd_prob),
            # transforms.RandShiftIntensityd(keys="image",
            #                                offsets=0.1,
            #                                prob=args.RandShiftIntensityd_prob)
    ]

    train_transforms = base_transforms + advanced_transforms

    if args.multilabel:
        train_transforms += [ConvertToMultiChannel(keys=['label'])]

    train_transforms += [transforms.ToTensord(keys=["image", "label"])]
    valid_transforms = base_transforms + [transforms.ToTensord(keys=["image", "label"])]

    train_transforms = transforms.Compose(train_transforms)
    valid_transforms = transforms.Compose(valid_transforms)

    if args.test_mode:
        transform = val_transforms
        test_ds = data.Dataset(data=val_files, transform=transform)
        test_sampler = Sampler(
            test_ds, shuffle=False)
        test_loader = data.DataLoader(test_ds,
                                      batch_size=1,
                                      shuffle=False,
                                      num_workers=args.num_workers,
                                      sampler=test_sampler,
                                      pin_memory=True,
                                      persistent_workers=True)
        loader = [test_loader, transform]
    else:
        if args.use_normal_dataset:
            train_ds = data.Dataset(
                data=train_files, transform=train_transforms)
        else:
            train_ds = data.CacheDataset(
                data=train_files,
                transform=train_transforms,
                # cache_num=24,
                cache_rate=1.0,
                num_workers=args.num_workers,
            )
        train_sampler = Sampler(train_ds)
        train_loader = data.DataLoader(train_ds,
                                       batch_size=args.batch_size_per_gpu,
                                       shuffle=(train_sampler is None),
                                       num_workers=args.num_workers,
                                       sampler=train_sampler,
                                       pin_memory=True,
                                       persistent_workers=True)
        # val_files = load_decathlon_datalist(datalist_json,
        #                                     True,
        #                                     "validation",
        #                                     base_dir=data_dir)
        val_ds = data.Dataset(data=val_files, transform=valid_transforms)
        val_sampler = Sampler(
            val_ds, shuffle=False)
        val_loader = data.DataLoader(val_ds,
                                     batch_size=1,
                                     shuffle=False,
                                     num_workers=args.num_workers,
                                     sampler=val_sampler,
                                     pin_memory=True,
                                     persistent_workers=True)
        loader = [train_loader, val_loader]

    return loader
