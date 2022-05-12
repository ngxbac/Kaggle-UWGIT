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
import cv2
from utils import dino as utils
import timm
from datasets.dataset25d import Dataset25D, get_transform
import segmentation_models_pytorch as smp
from criterions.segmentation import criterion_2d, dice_coef, iou_coef


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Train model 2.5D', add_help=False)

    # * Model
    # dataset parameters
    parser.add_argument('--csv', default='train_valid.csv')
    parser.add_argument('--input_size', default="320,384", type=str)
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--num_classes', default=3, type=int)
    parser.add_argument('--backbone', default='resnet34', type=str)

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
    parser.add_argument("--lr", default=2e-3, type=float, help="""Learning rate at the end of
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
    return parser


class Criterion(nn.Module):
    def __init__(
        self
    ):
        super().__init__()
        self.loss_fnc = criterion_2d

    def forward(self, logits, targets):
        ce_loss = self.loss_fnc(logits, targets)
        return ce_loss


class Metric:
    def __init__(self, num_classes=117):
        self.num_classes = num_classes

    def __call__(self, logits, targets):
        preds = logits.sigmoid()
        val_dice = dice_coef(targets, preds)
        val_jaccard = iou_coef(targets, preds)

        metric_dict = {
            'dice': val_dice,
            'iou': val_jaccard
        }

        return metric_dict


def get_dataset(args, name='train'):
    image_sizes = [int(x) for x in args.input_size.split(",")]
    train_transform = get_transform('train', image_sizes)
    valid_transform = get_transform('valid', image_sizes)

    if name == 'train':
        train_dataset = Dataset25D(
            csv=args.csv,
            label=True,
            is_train=True,
            fold=args.fold,
            transforms=train_transform
        )

        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset, shuffle=True)
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        print(f"Train data loaded: there are {len(train_dataset)} images.")
        return train_data_loader

    elif name == 'valid':
        valid_dataset = Dataset25D(
            csv=args.csv,
            label=True,
            is_train=False,
            fold=args.fold,
            transforms=valid_transform
        )

        valid_sampler = torch.utils.data.DistributedSampler(
            valid_dataset, shuffle=False)
        valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset,
            sampler=valid_sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        print(f"Valid data loaded: there are {len(valid_dataset)} images.")
        return valid_data_loader
    else:
        raise ("There is no dataset: ", name)


def get_model(args, distributed=True):
    model = smp.Unet(
        encoder_name=args.backbone,
        encoder_weights='imagenet',
        classes=args.num_classes
    )
    # move networks to gpu
    model = model.cuda()
    # synchronize batch norms (if any)
    if distributed:
        if utils.has_batchnorms(model):
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
    train_loader = get_dataset(args, name='train')
    valid_loader = get_dataset(args, name='valid')

    # ============ building Clusformer ... ============
    model = get_model(args)

    # ============ preparing loss ... ============
    criterion = Criterion()

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
            criterion=criterion,
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

        # ============ validate one epoch ... ============
        valid_stats = train_one_epoch(model, criterion,
                                      valid_loader, optimizer, scheduler, epoch, fp16_scaler, False, args)

        current_score = valid_stats['dice']
        if current_score > best_score:
            best_score = current_score
            is_save_best = True
        else:
            is_save_best = False

        valid_stats['best_score'] = best_score

        # ============ writing logs ... ============
        save_dict = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'criterion': criterion.state_dict(),
            'best_score': best_score
        }
        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(
            args.output_dir, 'checkpoint.pth'))
        if is_save_best:
            utils.save_on_master(save_dict, os.path.join(
                args.output_dir, 'best.pth'))
        if args.saveckp_freq and epoch % args.saveckp_freq == 0:
            utils.save_on_master(save_dict, os.path.join(
                args.output_dir, f'checkpoint{epoch:04}.pth'))
        log_train_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                           'epoch': epoch}
        log_valid_stats = {**{f'valid_{k}': v for k, v in valid_stats.items()},
                           'epoch': epoch}
        if utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_train_stats) + "\n")
                f.write(json.dumps(log_valid_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(model, criterion, data_loader, optimizer, scheduler, epoch, fp16_scaler, is_train, args):
    metric_fn = Metric(num_classes=args.num_classes)
    if is_train:
        model.train()
        prefix = 'TRAIN'
    else:
        model.eval()
        prefix = 'VALID'
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}/{}]'.format(epoch, args.epochs)
    for batch in metric_logger.log_every(data_loader, 50, header):
        # move images to gpu
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            if not is_train:
                with torch.no_grad():
                    logits = model(images)
            else:
                logits = model(images)
            loss = criterion(logits, targets)
            metric_dict = metric_fn(logits, targets)

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
        metric_logger.update(dice=metric_dict['dice'])
        metric_logger.update(iou=metric_dict['iou'])
        # for k, v in metric_dict.items():
        #     metric_logger.update(**{k: v})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print(f"[{prefix}] Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train 2.5D', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.pred:
        predict(args)
    else:
        train(args)
