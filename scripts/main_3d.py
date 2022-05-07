# Copyright 2020 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import logging
import os
import sys
import tempfile
from glob import glob
from sklearn.model_selection import KFold

import nibabel as nib
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import monai
from monai.data import create_test_image_3d, list_data_collate, decollate_batch
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.transforms import (
    Activations,
    AsChannelFirstd,
    RandSpatialCropd,
    Spacingd,
    AsDiscrete,
    Resized,
    Compose,
    LoadImaged,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    EnsureTyped,
    EnsureType,
    MapTransform,
    AddChanneld,
    NormalizeIntensityd,
    Orientationd,
    ResizeWithPadOrCropd
)
from monai.visualize import plot_2d_or_3d_image


class ConvertToMultiChanneld(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            result = []
            # merge label 2 and label 3 to construct TC
            result.append(d[key] == 1)
            result.append(d[key] == 2)
            result.append(d[key] == 3)
            result = np.concatenate(result, axis=0).astype(np.float32)
            d[key] = result
        return d


class DebugTransform(MapTransform):
    def __call__(self, data):
        d = dict(data)
        for key in self.keys:
            print(d[key].shape)
            # result = []
            # # merge label 2 and label 3 to construct TC
            # result.append(d[key] == 1)
            # result.append(d[key] == 2)
            # result.append(d[key] == 3)
            # result = np.concatenate(result, axis=0).astype(np.float32)
            # d[key] = result
        return d


def main(tempdir):
    monai.config.print_config()
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    # failed = ['case7_day0', 'case81_day30', 'case85_day27']

    # def is_failed(path):
    #     for f in failed:
    #         if f in path:
    #             return True

    #     return False

    train_fold = 0

    images = sorted(glob(os.path.join(tempdir, "train/*/*.nii.gz")))

    all_cases = [image.split("_")[-2] for image in images]
    unique_cases = np.unique(all_cases)
    kf = KFold(n_splits=5, random_state=2411, shuffle=True)
    for fold, (train_idx, valid_idx) in enumerate(kf.split(unique_cases)):
        if fold == train_fold:
            train_cases = unique_cases[train_idx]
            valid_cases = unique_cases[valid_idx]

    train_images = [image for image in images if image.split(
        "/")[-2] in train_cases]
    train_segs = [image.replace('train', 'mask') for image in train_images]

    val_images = [image for image in images if image.split(
        "/")[-2] in valid_cases]
    val_segs = [image.replace('train', 'mask') for image in val_images]

    train_files = [{"img": img, "seg": seg}
                   for img, seg in zip(train_images, train_segs)]
    val_files = [{"img": img, "seg": seg}
                 for img, seg in zip(val_images, val_segs)]

    roi_size = [80, 80, 80]

    # define transforms for image and segmentation
    train_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            # DebugTransform(keys=['img', 'seg']),
            AddChanneld(keys=['img', 'seg']),
            # Orientationd(keys=["img", "seg"], axcodes="RAS"),
            # Spacingd(
            #     keys=["img", "seg"],
            #     pixdim=(1.0, 1.0, 1.0),
            #     mode=("bilinear", "nearest"),
            # ),

            # ScaleIntensityd(keys="img"),
            RandCropByPosNegLabeld(
                keys=["img", "seg"], label_key="seg", spatial_size=roi_size, pos=1, neg=1, num_samples=4
            ),
            # RandSpatialCropd(keys=["img", "seg"],
            #                  roi_size=roi_size, random_size=False),
            NormalizeIntensityd(keys=["img"], nonzero=True, channel_wise=True),
            ResizeWithPadOrCropd(keys=['img', 'seg'], spatial_size=roi_size),
            # Resized(keys=['img', 'seg'], spatial_size=[
            #         64, 64, 64], mode=('area', 'nearest')),
            RandRotate90d(keys=["img", "seg"], prob=0.5, spatial_axes=[0, 2]),
            ConvertToMultiChanneld(keys=['seg']),
            EnsureTyped(keys=["img", "seg"]),
        ]
    )
    val_transforms = Compose(
        [
            LoadImaged(keys=["img", "seg"]),
            AddChanneld(keys=['img', 'seg']),
            # Orientationd(keys=["img", "seg"], axcodes="RAS"),
            # Spacingd(
            #     keys=["img", "seg"],
            #     pixdim=(1.0, 1.0, 1.0),
            #     mode=("bilinear", "nearest"),
            # ),
            # AsChannelFirstd(keys=["img", "seg"], channel_dim=-1),
            # ScaleIntensityd(keys="img"),
            NormalizeIntensityd(
                keys=["img"], nonzero=True, channel_wise=True),
            ConvertToMultiChanneld(keys=['seg']),
            EnsureTyped(keys=["img", "seg"]),
        ]
    )

    # define dataset, data loader
    # check_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    # check_loader = DataLoader(check_ds, batch_size=2,
    #                           num_workers=4, collate_fn=list_data_collate)
    # check_data = monai.utils.misc.first(check_loader)
    # print(check_data["img"].shape, check_data["seg"].shape)

    # create a training data loader
    train_ds = monai.data.Dataset(data=train_files, transform=train_transforms)
    # use batch_size=2 to load images and use RandCropByPosNegLabeld to generate 2 x 4 images for network training
    train_loader = DataLoader(
        train_ds,
        batch_size=4,
        shuffle=True,
        num_workers=4,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),
    )
    # create a validation data loader
    val_ds = monai.data.Dataset(data=val_files, transform=val_transforms)
    val_loader = DataLoader(val_ds, batch_size=1,
                            num_workers=4, collate_fn=list_data_collate)
    dice_metric = DiceMetric(include_background=True,
                             reduction="mean", get_not_nans=False)
    post_trans = Compose([EnsureType(), Activations(
        sigmoid=True), AsDiscrete(threshold=0.5)])
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = monai.networks.nets.UNet(
    #     spatial_dims=3,
    #     in_channels=1,
    #     out_channels=3,
    #     channels=(16, 32, 64, 128, 256),
    #     strides=(2, 2, 2, 2),
    #     num_res_units=2,
    # ).to(device)

    # model = monai.networks.nets.SegResNet(
    #     blocks_down=[1, 2, 2, 4],
    #     blocks_up=[1, 1, 1],
    #     init_filters=16,
    #     in_channels=1,
    #     out_channels=3,
    #     dropout_prob=0.2,
    # ).to(device)

    model = monai.networks.nets.UNETR(
        in_channels=1, out_channels=3, img_size=roi_size,
        feature_size=32
    ).to(device)
    loss_function = monai.losses.DiceLoss(
        smooth_nr=0, smooth_dr=1e-5, squared_pred=True, to_onehot_y=False, sigmoid=True)
    optimizer = torch.optim.Adam(model.parameters(), 1e-3)

    num_epochs = 300
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs * len(train_loader))

    # start a typical PyTorch training
    val_interval = 1
    best_metric = -1
    best_metric_epoch = -1
    epoch_loss_values = list()
    metric_values = list()
    writer = SummaryWriter()
    for epoch in range(num_epochs):
        print("-" * 10)
        print(f"epoch {epoch + 1}/{num_epochs}")
        model.train()
        epoch_loss = 0
        step = 0
        for batch_data in train_loader:
            step += 1
            inputs, labels = batch_data["img"].to(
                device), batch_data["seg"].to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            epoch_len = len(train_ds) // train_loader.batch_size
            lr_scheduler.step()
            current_lr = optimizer.param_groups[0]["lr"]
            print(
                f"{step}/{epoch_len}, train_loss: {loss.item():.4f}, lr: {current_lr:.4f}")
            writer.add_scalar("train_loss", loss.item(),
                              epoch_len * epoch + step)
        epoch_loss /= step
        epoch_loss_values.append(epoch_loss)
        print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")

        if (epoch + 1) % val_interval == 0:
            model.eval()
            with torch.no_grad():
                val_images = None
                val_labels = None
                val_outputs = None
                for val_data in val_loader:
                    val_images, val_labels = val_data["img"].to(
                        device), val_data["seg"].to(device)
                    # roi_size = (64, 112, 112)
                    sw_batch_size = 4
                    val_outputs = sliding_window_inference(
                        val_images, roi_size, sw_batch_size, model)
                    # import pdb
                    # pdb.set_trace()
                    val_outputs = [post_trans(i)
                                   for i in decollate_batch(val_outputs)]
                    # compute metric for current iteration
                    dice_metric(y_pred=val_outputs, y=val_labels)
                # aggregate the final mean dice result
                metric = dice_metric.aggregate().item()
                # reset the status for next validation round
                dice_metric.reset()

                metric_values.append(metric)
                if metric > best_metric:
                    best_metric = metric
                    best_metric_epoch = epoch + 1
                    torch.save(model.state_dict(),
                               "best_metric_model_segmentation3d_dict.pth")
                    print("saved new best metric model")
                print(
                    "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                        epoch + 1, metric, best_metric, best_metric_epoch
                    )
                )
                writer.add_scalar("val_mean_dice", metric, epoch + 1)
                # plot the last model output as GIF image in TensorBoard with the corresponding image and label
                plot_2d_or_3d_image(val_images, epoch + 1,
                                    writer, index=0, tag="image")
                plot_2d_or_3d_image(val_labels, epoch + 1,
                                    writer, index=0, tag="label")
                plot_2d_or_3d_image(val_outputs, epoch + 1,
                                    writer, index=0, tag="output")

    print(
        f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
    writer.close()


if __name__ == "__main__":
    main(tempdir='data/nii-data/')
