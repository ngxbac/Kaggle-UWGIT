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


from monai.transforms import (
    Compose,
    LoadImaged,
    RandSpatialCropd,
    EnsureTyped,
    CastToTyped,
    NormalizeIntensityd,
    RandFlipd,
    CenterSpatialCropd,
    ScaleIntensityRanged,
    RandAffined,
    RandScaleIntensityd,
    RandShiftIntensityd,
    RandCoarseDropoutd,
    Rand2DElasticd,
    Lambdad,
    Resized,
    AddChanneld,
    RandGaussianNoised,
    RandGridDistortiond,
    RepeatChanneld,
    Transposed,
    OneOf,
    EnsureChannelFirstd,
    RandLambdad,
    Spacingd,
    FgBgToIndicesd,
    CropForegroundd,
    RandCropByPosNegLabeld,
    ToDeviced,
    SpatialPadd,

)


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


def get_loader(args):
    import json
    fold = args.fold
    data_dir = args.data_dir

    json_file = f'{data_dir}/dataset_3d_fold_{fold}.json'
    print(json_file)

    with open(json_file, 'r') as f:
        data_json = json.load(f)

    train_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            # Spacingd(keys=["image", "mask"], pixdim=cfg.spacing, mode="nearest"),
            Lambdad(keys="image", func=lambda x: x / x.max()),
            # SpatialPadd(keys=("image", "mask"), spatial_size=cfg.img_size),

            transforms.RandCropByPosNegLabeld(
                keys=["image", "mask"],
                label_key="mask",
                spatial_size=(args.roi_x, args.roi_y, args.roi_z),
                pos=1,
                neg=1,
                num_samples=args.num_samples,
                image_key="image",
            ),

            # RandSpatialCropd(
            #     keys=("image", "mask"),
            #     roi_size=(args.roi_x, args.roi_y, args.roi_z),
            #     random_size=False,
            # ),
            RandFlipd(keys=("image", "mask"), prob=0.5, spatial_axis=[0]),
            RandFlipd(keys=("image", "mask"), prob=0.5, spatial_axis=[1]),
            # RandFlipd(keys=("image", "mask"), prob=0.5, spatial_axis=[2]),
            RandAffined(
                keys=("image", "mask"),
                prob=0.5,
                rotate_range=np.pi / 12,
                translate_range=(args.roi_x*0.0625, args.roi_y*0.0625),
                scale_range=(0.1, 0.1),
                mode="nearest",
            ),
            OneOf(
                [
                    RandGridDistortiond(keys=("image", "mask"), prob=0.5, distort_limit=(-0.05, 0.05), mode="nearest"),
                    RandCoarseDropoutd(
                        keys=("image", "mask"),
                        holes=5,
                        max_holes=8,
                        spatial_size=(1, 1, 1),
                        max_spatial_size=(12, 12, 12),
                        fill_value=0.0,
                        prob=0.5,
                    ),
                ]
            ),
            RandScaleIntensityd(keys="image", factors=(-0.2, 0.2), prob=0.5),
            RandShiftIntensityd(keys="image", offsets=(-0.1, 0.1), prob=0.5),
            EnsureTyped(keys=("image", "mask"), dtype=torch.float32),
        ]
    )

    val_transforms = Compose(
        [
            LoadImaged(keys=["image", "mask"]),
            EnsureChannelFirstd(keys=["image", "mask"]),
            # Spacingd(keys=["image", "mask"], pixdim=cfg.spacing, mode="nearest"),
            Lambdad(keys="image", func=lambda x: x / x.max()),
            EnsureTyped(keys=("image", "mask"), dtype=torch.float32),
        ]
    )


    train_ds = data.CacheDataset(
        data=data_json['train'],
        transform=train_transforms,
        cache_num=24,
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
    val_ds = data.CacheDataset(
        data=data_json["val"],
        transform=val_transforms,
        cache_rate=1.0,
        num_workers=args.num_workers,
        copy_cache=False,
    )
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
