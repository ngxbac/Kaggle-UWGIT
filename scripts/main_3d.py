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

import nibabel as nib
import numpy as np
import torch


import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
from utils import dino as utils
from datasets.dataset3d import get_loader
from functools import partial
import timm

from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss, DiceCELoss, DiceFocalLoss
from monai.metrics import DiceMetric, HausdorffDistanceMetric
from monai.utils.enums import MetricReduction
from monai.transforms import AsDiscrete, Activations, Compose, EnsureType
from monai.networks.nets import DynUNet, SegResNet, UNETR
from monai.data import decollate_batch


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Train model 2.5D', add_help=False)

    # * Model
    # dataset parameters
    parser.add_argument('--data_dir', default='/dataset/dataset0/',
                        type=str, help='dataset directory')
    parser.add_argument('--fold', default=0, type=int)

    # Settings
    parser.add_argument('--model_name', default='unetr',
                        type=str, help='model name')
    parser.add_argument('--pos_embed', default='perceptron',
                        type=str, help='type of position embedding')
    parser.add_argument('--norm_name', default='instance',
                        type=str, help='normalization layer type in decoder')
    parser.add_argument('--num_heads', default=12, type=int,
                        help='number of attention heads in ViT encoder')
    parser.add_argument('--mlp_dim', default=3072, type=int,
                        help='mlp dimention in ViT encoder')
    parser.add_argument('--hidden_size', default=768, type=int,
                        help='hidden size dimention in ViT encoder')
    parser.add_argument('--feature_size', default=16, type=int,
                        help='feature size dimention')
    parser.add_argument('--in_channels', default=1, type=int,
                        help='number of input channels')
    parser.add_argument('--out_channels', default=4, type=int,
                        help='number of output channels')
    parser.add_argument('--res_block', action='store_true',
                        help='use residual blocks')
    parser.add_argument('--conv_block', action='store_true',
                        help='use conv blocks')
    parser.add_argument('--use_normal_dataset',
                        action='store_true', help='use monai Dataset class')
    parser.add_argument('--a_min', default=0, type=float,
                        help='a_min in ScaleIntensityRanged')
    parser.add_argument('--a_max', default=750.0, type=float,
                        help='a_max in ScaleIntensityRanged')
    parser.add_argument('--b_min', default=0.0, type=float,
                        help='b_min in ScaleIntensityRanged')
    parser.add_argument('--b_max', default=1.0, type=float,
                        help='b_max in ScaleIntensityRanged')
    parser.add_argument('--space_x', default=1.5, type=float,
                        help='spacing in x direction')
    parser.add_argument('--space_y', default=1.5, type=float,
                        help='spacing in y direction')
    parser.add_argument('--space_z', default=1.5, type=float,
                        help='spacing in z direction')
    parser.add_argument('--roi_x', default=96, type=int,
                        help='roi size in x direction')
    parser.add_argument('--roi_y', default=96, type=int,
                        help='roi size in y direction')
    parser.add_argument('--roi_z', default=96, type=int,
                        help='roi size in z direction')
    parser.add_argument('--dropout_rate', default=0.0,
                        type=float, help='dropout rate')
    parser.add_argument('--RandFlipd_prob', default=0.2,
                        type=float, help='RandFlipd aug probability')
    parser.add_argument('--RandRotate90d_prob', default=0.2,
                        type=float, help='RandRotate90d aug probability')
    parser.add_argument('--RandScaleIntensityd_prob', default=0.1,
                        type=float, help='RandScaleIntensityd aug probability')
    parser.add_argument('--RandShiftIntensityd_prob', default=0.1,
                        type=float, help='RandShiftIntensityd aug probability')
    parser.add_argument('--infer_overlap', default=0.5, type=float,
                        help='sliding window inference overlap')
    parser.add_argument('--resume_ckpt', action='store_true',
                        help='resume training from pretrained checkpoint')
    parser.add_argument('--resume_jit', action='store_true',
                        help='resume training from pretrained torchscript checkpoint')
    parser.add_argument('--smooth_dr', default=1e-6, type=float,
                        help='constant added to dice denominator to avoid nan')
    parser.add_argument('--smooth_nr', default=0.0, type=float,
                        help='constant added to dice numerator to avoid zero')
    parser.add_argument('--num_samples', default=32, type=int,
                        help='Number of samples')

    # Training/Optimization parameters
    parser.add_argument('--use_fp16', type=utils.bool_flag, default=True, help="""Whether or not
        to use half precision for training. Improves training time and memory requirements,
        but can provoke instability and slight decay of performance. We recommend disabling
        mixed precision if the loss is unstable, if reducing the patch size or if training with bigger ViTs.""")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="""Initial value of the
        weight decay.""")
    parser.add_argument('--batch_size_per_gpu', default=128, type=int,
                        help='Per-GPU batch-size : number of distinct images loaded on one GPU.')
    parser.add_argument('--epochs', default=20, type=int,
                        help='Number of epochs of training.')
    parser.add_argument("--lr", default=1e-3, type=float, help="""Learning rate at the end of
        linear warmup (highest LR used during training). The learning rate is linearly scaled
        with the batch size, and specified here for a reference batch size of 256.""")

    # Misc
    parser.add_argument('--output_dir', default=".", type=str,
                        help='Path to save logs and checkpoints.')
    parser.add_argument('--saveckp_freq', default=5, type=int,
                        help='Save checkpoint every x epochs.')
    parser.add_argument('--seed', default=216, type=int, help='Random seed.')
    parser.add_argument('--num_workers', default=4, type=int,
                        help='Number of data loading workers per GPU.')
    parser.add_argument("--dist_url", default="env://", type=str, help="""url used to set up
        distributed training; see https://pytorch.org/docs/stable/distributed.html""")
    parser.add_argument("--local_rank", default=0, type=int,
                        help="Please ignore and do not set this argument.")
    parser.add_argument('--pred', type=utils.bool_flag, default=False)
    parser.add_argument('--resume', type=utils.bool_flag, default=False)
    parser.add_argument('--test_mode', type=utils.bool_flag, default=False)
    return parser


def get_model(args):
    # pretrained_dir = args.pretrained_dir
    if (args.model_name is None) or args.model_name == 'unetr':
        model = UNETR(
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            feature_size=args.feature_size,
            hidden_size=args.hidden_size,
            mlp_dim=args.mlp_dim,
            num_heads=args.num_heads,
            pos_embed=args.pos_embed,
            norm_name=args.norm_name,
            conv_block=True,
            res_block=True,
            dropout_rate=args.dropout_rate)

    elif args.model_name == 'dynunet':
        def get_kernels_strides(args):
            """
            This function is only used for decathlon datasets with the provided patch sizes.
            When refering this method for other tasks, please ensure that the patch size for each spatial dimension should
            be divisible by the product of all strides in the corresponding dimension.
            In addition, the minimal spatial size should have at least one dimension that has twice the size of
            the product of all strides. For patch sizes that cannot find suitable strides, an error will be raised.
            """
            sizes, spacings = [args.roi_x, args.roi_y,
                               args.roi_z], [args.space_x, args.space_y, args.space_z]
            input_size = sizes
            strides, kernels = [], []
            while True:
                spacing_ratio = [sp / min(spacings) for sp in spacings]
                stride = [
                    2 if ratio <= 2 and size >= 8 else 1
                    for (ratio, size) in zip(spacing_ratio, sizes)
                ]
                kernel = [3 if ratio <= 2 else 1 for ratio in spacing_ratio]
                if all(s == 1 for s in stride):
                    break
                for idx, (i, j) in enumerate(zip(sizes, stride)):
                    if i % j != 0:
                        raise ValueError(
                            f"Patch size is not supported, please try to modify the size {input_size[idx]} in the spatial dimension {idx}."
                        )
                sizes = [i / j for i, j in zip(sizes, stride)]
                spacings = [i * j for i, j in zip(spacings, stride)]
                kernels.append(kernel)
                strides.append(stride)

            strides.insert(0, len(spacings) * [1])
            kernels.append(len(spacings) * [3])
            return kernels, strides

        kernels, strides = get_kernels_strides(args)
        model = DynUNet(
            spatial_dims=3,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            kernel_size=kernels,
            strides=strides,
            upsample_kernel_size=strides[1:],
            norm_name=args.norm_name,
            deep_supervision=False
            # deep_supr_num=1
        )

    elif args.model_name == 'segresnet':
        model = SegResNet(
            blocks_down=[1, 2, 2, 4],
            blocks_up=[1, 1, 1],
            init_filters=16,
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            dropout_prob=0.2,
        )
    else:
        raise ValueError('Unsupported model ' + str(args.model_name))

    # if args.resume_ckpt:
    #     model_dict = torch.load(os.path.join(
    #         pretrained_dir, args.pretrained_model_name))
    #     model.load_state_dict(model_dict)
    #     print('Use pretrained weights')

    # if args.resume_jit:
    #     if not args.noamp:
    #         print(
    #             'Training from pre-trained checkpoint does not support AMP\nAMP is disabled.')
    #         args.amp = args.noamp
    #     model = torch.jit.load(os.path.join(
    #         pretrained_dir, args.pretrained_model_name))

    # move networks to gpu
    model = model.cuda()
    # synchronize batch norms (if any)
    if args.norm_name == 'batch' and utils.has_batchnorms(model):
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model = nn.parallel.DistributedDataParallel(
        model, device_ids=[args.gpu], find_unused_parameters=True)

    return model


def train(args):
    utils.init_distributed_mode(args)
    utils.fix_random_seeds(args.seed)
    # lr = (args.lr / 64) * torch.distributed.get_world_size() * \
    #     args.batch_size_per_gpu

    # args.lr = lr
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v))
          for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    train_loader, valid_loader = get_loader(args)
    # ============ building Clusformer ... ============
    model = get_model(args)

    # ============ preparing loss ... ============
    criterion = DiceFocalLoss(to_onehot_y=False,
                              sigmoid=True,
                              squared_pred=False,
                              smooth_nr=args.smooth_nr,
                              smooth_dr=args.smooth_dr)

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=len(train_loader) * args.epochs, eta_min=0)
    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0}
    best_score = -np.inf
    is_save_best = False
    if args.resume:
        utils.restart_from_checkpoint(
            os.path.join(args.output_dir, "checkpoint.pth"),
            run_variables=to_restore,
            model=model,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            scheduler=scheduler,
            best_score=best_score
        )

    start_epoch = to_restore["epoch"]

    start_time = time.time()
    print("Starting eyestate training !")
    for epoch in range(start_epoch, args.epochs):
        train_loader.sampler.set_epoch(epoch)

        # ============ training one epoch ... ============
        train_stats = train_one_epoch(model, criterion,
                                      train_loader, optimizer, scheduler, epoch, fp16_scaler, True, args)

        # Distributed bn
        timm.utils.distribute_bn(
            model, torch.distributed.get_world_size(), True)

        save_dict = {
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'criterion': criterion.state_dict(),
        }

        if epoch % 5 == 0:
            # ============ validate one epoch ... ============
            valid_stats = valid_one_epoch(
                model, valid_loader, epoch, fp16_scaler, args)

            current_score = valid_stats['dice']
            if current_score > best_score:
                best_score = current_score
                is_save_best = True
            else:
                is_save_best = False

            valid_stats['best_score'] = best_score

            # ============ writing logs ... ============
            if fp16_scaler is not None:
                save_dict['fp16_scaler'] = fp16_scaler.state_dict()
            utils.save_on_master(save_dict, os.path.join(
                args.output_dir, 'checkpoint.pth'))
            if is_save_best:
                utils.save_on_master(save_dict, os.path.join(
                    args.output_dir, 'best.pth'))

            log_valid_stats = {**{f'valid_{k}': v for k, v in valid_stats.items()},
                               'epoch': epoch}

            if utils.is_main_process():
                with (Path(args.output_dir) / "log.txt").open("a") as f:
                    f.write(json.dumps(log_valid_stats) + "\n")

        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(
                args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_train_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                           'epoch': epoch}

        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_train_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model, criterion, data_loader, optimizer, scheduler, epoch, fp16_scaler, is_train, args):
    model.train()
    prefix = "TRAIN"
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for batch in metric_logger.log_every(data_loader, 50, header):
        # move images to gpu
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['label'].cuda(non_blocking=True)

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            if not is_train:
                with torch.no_grad():
                    logits = model(images)
            else:
                logits = model(images)
            loss = criterion(logits, targets)
            # metric_dict = metric_fn(logits, targets)

        if not math.isfinite(loss.item()):
            print("Loss is {}, stopping training".format(loss.item()), force=True)
            sys.exit(1)

        if is_train:
            # student update
            optimizer.zero_grad()
            param_norms = None
            if fp16_scaler is None:
                loss.backward()
                optimizer.step()
            else:
                fp16_scaler.scale(loss).backward()
                fp16_scaler.step(optimizer)
                fp16_scaler.update()

            scheduler.step()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"[{prefix}] Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def valid_one_epoch(model, data_loader, epoch, fp16_scaler, args):
    inf_size = [args.roi_x, args.roi_y, args.roi_z]
    model.eval()

    # Functions for post evaluation

    # post_label = AsDiscrete(to_onehot=True,
    #                         n_classes=args.out_channels)

    # post_pred = AsDiscrete(argmax=True,
    #                        to_onehot=True,
    #                        n_classes=args.out_channels)

    def post_label(x):
        return x

    post_pred = Compose(
        [EnsureType(), Activations(sigmoid=True), AsDiscrete(threshold=0.5)]
    )

    metric_fn = DiceMetric(include_background=True,
                           reduction=MetricReduction.MEAN,
                           get_not_nans=True)

    model_inferer = partial(sliding_window_inference,
                            roi_size=inf_size,
                            sw_batch_size=1,
                            predictor=model,
                            overlap=args.infer_overlap)

    prefix = 'VALID'
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for batch in metric_logger.log_every(data_loader, 1, header):

        # move images to gpu
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['label'].cuda(non_blocking=True)

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            with torch.no_grad():
                logits = model_inferer(images)

            val_labels_list = decollate_batch(targets)
            val_labels_convert = [post_label(
                val_label_tensor) for val_label_tensor in val_labels_list]
            val_outputs_list = decollate_batch(logits)
            val_output_convert = [post_pred(val_pred_tensor)
                                  for val_pred_tensor in val_outputs_list]
            acc = metric_fn(y_pred=val_output_convert, y=val_labels_convert)
            acc = acc.mean()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(dice=acc.item())
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"[{prefix}] Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train 3D', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    train(args)
