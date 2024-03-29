import argparse
import os
import sys
sys.path.append(".")
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
from datasets.uw_gi import UWGI, get_transform
from datasets.abdomen import Abdomen
from datasets.chaos import CHAOS
import segmentation_models_pytorch as smp
from criterions.segmentation import criterion_2d, dice_coef, iou_coef, ComboLoss
from schedulers import OneCycleLRWithWarmup


def get_args_parser():
    parser = argparse.ArgumentParser(
        'Train model 2.5D', add_help=False)

    # * Model
    # dataset parameters
    parser.add_argument('--csv', default='train_valid.csv')
    parser.add_argument('--data_dir', default='data/uw-gi-25d/')
    parser.add_argument('--input_size', default="512,512", type=str)
    parser.add_argument('--fold', default=0, type=int)
    parser.add_argument('--num_classes', default=4, type=int)
    parser.add_argument('--model_name', default='FPN', type=str)
    parser.add_argument('--backbone', default='timm-efficientnet-b5', type=str)
    parser.add_argument('--loss_weights', type=str, default='1,0,0')
    parser.add_argument('--pretrained_checkpoint', type=str, default='')
    parser.add_argument('--multilabel', type=utils.bool_flag, default=False)
    parser.add_argument('--use_ema', type=utils.bool_flag, default=False)
    parser.add_argument('--ema_decay', default=0.997, type=float)
    parser.add_argument('--pretrained', type=utils.bool_flag, default=True)
    parser.add_argument('--aux', type=utils.bool_flag, default=False)
    parser.add_argument('--dataset', default="uw-gi", type=str)
    parser.add_argument('--mmcfg', default='', type=str)
    parser.add_argument('--vit_patches_size', default=16, type=int)

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
    parser.add_argument("--min_lr", default=0.0, type=float)
    parser.add_argument('--scheduler', default='cosine', type=str)

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
    parser.add_argument('--pseudo', type=utils.bool_flag, default=False)
    parser.add_argument('--resume', type=str, default='')
    return parser


class Criterion(nn.Module):
    def __init__(
        self,
        args
    ):
        super().__init__()
        loss_weights = args.loss_weights
        loss_weights = [int(x) for x in loss_weights.split(",")]
        self.seg_loss = ComboLoss(
            args.multilabel,
            loss_weights[0],
            loss_weights[1],
            loss_weights[2]
        )

        self.aux = args.aux

        self.aux_loss = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        if args.aux:
            masks, auxs = logits
            gt_masks, gt_auxs = targets
            seg_loss = self.seg_loss(masks, gt_masks)
            aux_loss = self.aux_loss(auxs, gt_auxs)
            return (seg_loss + aux_loss) / 2
        else:
            return self.seg_loss(logits, targets)


class Metric:
    def __init__(self, multilabel=True, aux=False):
        self.multilabel = multilabel
        self.aux = aux

    def multilabel_metric(self, logits, targets):
        preds = logits.sigmoid()
        val_dice = dice_coef(targets, preds)

        metric_dict = {
            'dice': val_dice,
        }

        return metric_dict

    def measure_dice(self, y_pred, y_true, dim=(1, 2), epsilon=1e-8):
        inter = (y_true*y_pred).sum(dim=dim)
        den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
        dice = ((2*inter+epsilon)/(den+epsilon)).mean()
        return dice

    def multiclass_metric(self, logits, targets):
        # logits: B x C x H x W
        # targets: B x H x W
        num_classes = logits.shape[1]
        preds = logits.argmax(dim=1)  # B x H x W

        all_dices = []
        for i in range(1, num_classes):
            pred_cls = preds == i
            target_cls = targets == i
            dice_cls = self.measure_dice(pred_cls, target_cls)
            all_dices.append(dice_cls.item())

        metric_dict = {
            'dice': np.mean(all_dices)
        }

        return metric_dict

    def __call__(self, logits, targets):
        ret_dict = {}

        if self.aux:
            masks, auxs = logits
            gt_masks, gt_auxs = targets
            f1_score = utils.f1_score(
                auxs.sigmoid() > 0.5,
                gt_auxs
            )
            ret_dict['f1'] = f1_score
        else:
            masks = logits
            gt_masks = targets

        if self.multilabel:
            ret = self.multilabel_metric(masks, gt_masks)
        else:
            ret = self.multiclass_metric(masks, gt_masks)

        ret_dict.update(ret)
        return ret_dict


def get_abdomen_dataset(args):
    image_sizes = [int(x) for x in args.input_size.split(",")]
    train_transform = get_transform('train', image_sizes)
    data_dirs = args.data_dir
    data_dirs = data_dirs.split(",")

    train_dataset = [Abdomen(
        data_dir=data_dir,
        transforms=train_transform
    ) for data_dir in data_dirs]

    train_dataset = torch.utils.data.ConcatDataset(train_dataset)

    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset, shuffle=True) if args.distributed else None
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        shuffle=train_sampler is None,
    )
    print(f"Train data loaded: there are {len(train_dataset)} images.")
    return train_data_loader


def get_chaos_dataset(args, name='train'):
    image_sizes = [int(x) for x in args.input_size.split(",")]
    train_transform = get_transform('train', image_sizes)
    valid_transform = get_transform('valid', image_sizes)

    if name == 'train':
        transform = train_transform
        is_test = False
    else:
        transform = valid_transform
        is_test = True

    train_dataset = CHAOS(transform, is_test)

    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset, shuffle=True) if args.distributed else None
    train_data_loader = torch.utils.data.DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=args.batch_size_per_gpu,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=train_sampler is None,
    )
    print(f"Train data loaded: there are {len(train_dataset)} images.")
    return train_data_loader


def get_uw_gi_dataset(args, name='train'):
    image_sizes = [int(x) for x in args.input_size.split(",")]
    train_transform = get_transform('train', image_sizes)
    valid_transform = get_transform('valid', image_sizes)

    if name == 'train':
        train_dataset = UWGI(
            csv=args.csv,
            data_dir=args.data_dir,
            is_train=True,
            multilabel=args.multilabel,
            fold=args.fold,
            transforms=train_transform,
            infer_pseudo=args.pseudo
        )

        # chaos_dataset = CHAOS(train_transform, False)
        # train_dataset = torch.utils.data.ConcatDataset([train_dataset, chaos_dataset])

        train_sampler = torch.utils.data.DistributedSampler(
            train_dataset, shuffle=True) if args.distributed else None
        train_data_loader = torch.utils.data.DataLoader(
            train_dataset,
            sampler=train_sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
            shuffle=train_sampler is None,
        )
        print(f"Train data loaded: there are {len(train_dataset)} images.")
        return train_data_loader

    elif name == 'valid':
        valid_dataset = UWGI(
            csv=args.csv,
            data_dir=args.data_dir,
            is_train=False,
            multilabel=args.multilabel,
            fold=args.fold,
            transforms=valid_transform,
            infer_pseudo=args.pseudo
        )

        valid_sampler = None if (args.pred or args.pseudo or not args.distributed) else torch.utils.data.DistributedSampler(
            valid_dataset, shuffle=False)
        valid_data_loader = torch.utils.data.DataLoader(
            valid_dataset,
            sampler=valid_sampler,
            batch_size=args.batch_size_per_gpu,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
            shuffle=False
        )
        print(f"Valid data loaded: there are {len(valid_dataset)} images.")
        return valid_data_loader
    else:
        raise ("There is no dataset: ", name)


def get_dataset(args, name='train'):
    if args.dataset == 'uw-gi':
        return get_uw_gi_dataset(args, name)
    elif args.dataset == 'chaos':
        return get_chaos_dataset(args, name)
    elif args.dataset == 'abdomen':
        return get_abdomen_dataset(args)


def load_pretrained_checkpoint(model, path):
    current_state_dict = model.state_dict()
    checkpoint = torch.load(path, map_location='cpu')
    if 'ema' in checkpoint:
        print("[+] Load EMA weight")
        checkpoint = checkpoint['ema']
    elif 'model' in checkpoint:
        print("[+] Load model weight")
        checkpoint = checkpoint['model']

    for k in current_state_dict.keys():
        if current_state_dict[k].shape == checkpoint[k].shape:
            current_state_dict[k] = checkpoint[k]
        else:
            print(f"{k} is not matched")

    model.load_state_dict(current_state_dict)
    print(f'[+] Loaded checkpoint: {path}')
    return model


def get_model(args, distributed=True):

    if args.model_name == 'segformer':
        from transformers import SegformerForSemanticSegmentation
        model = SegformerForSemanticSegmentation.from_pretrained(
            f"nvidia/mit-{args.backbone}",
            num_labels=args.num_classes
        )
    elif 'mmseg' in args.model_name and os.path.isfile(args.mmcfg):
        from models.mmmodels import get_mmseg_models
        model = get_mmseg_models(args)
    else:
        if args.aux:
            aux_params = dict(
                pooling='avg',             # one of 'avg', 'max'
                dropout=0.5,               # dropout ratio, default is None
                activation=None,      # activation function, default is None
                classes=1,                 # define number of output labels
            )
        else:
            aux_params = None
        encoder_weights = 'imagenet'
        encoder_depth = 5

        decoder_channels = (1024, 512, 256, 128, 64)
        if 'timm-efficientnet' in args.backbone:
            encoder_weights = 'noisy-student'
        elif 'convnext' in args.backbone:
            encoder_weights = 'imagenet'
            encoder_depth = 4

        model = smp.__dict__[args.model_name](
            encoder_name=args.backbone,
            encoder_weights=encoder_weights,
            classes=args.num_classes,
            in_channels=3,
            encoder_depth=encoder_depth,
            aux_params=aux_params
        )

    print(model)

    # move networks to gpu
    model = model.cuda()
    if args.use_ema:
        model_ema = timm.utils.ModelEmaV2(model, decay=args.ema_decay)
    else:
        model_ema = None

    # synchronize batch norms (if any)
    if distributed:
        if utils.has_batchnorms(model):
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], find_unused_parameters=True)
    else:
        model = nn.DataParallel(model)

    if os.path.isfile(args.pretrained_checkpoint):
        model = load_pretrained_checkpoint(model, args.pretrained_checkpoint)

    return model, model_ema


def train(args):
    # args.distributed = hasattr(args, 'gpu')
    args.distributed = True

    if "WORLD_SIZE" in os.environ:
        args.world_size = int(os.environ["WORLD_SIZE"])

    if args.distributed:
        utils.init_distributed_mode(args)

    utils.fix_random_seeds(args.seed)
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v))
          for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    train_loader = get_dataset(args, name='train')
    if args.dataset == 'uw-gi':
        valid_loader = get_dataset(args, name='valid')
    else:
        valid_loader = None

    # ============ building Clusformer ... ============
    model, model_ema = get_model(args, args.distributed)

    # ============ preparing loss ... ============
    # loss_weights = args.loss_weights
    # loss_weights = [int(x) for x in loss_weights.split(",")]
    criterion = Criterion(
        args
    )

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)

    total_steps = args.epochs * len(train_loader)
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps, eta_min=args.min_lr)
    elif args.scheduler == 'onecycle':
        scheduler = OneCycleLRWithWarmup(
            optimizer=optimizer,
            num_steps=total_steps,
            lr_range=(args.lr, args.lr / 10),
            warmup_fraction=0.1,
            init_lr=args.lr / 10
        )
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', patience=2, factor=0.1
        )

    # for mixed precision training
    fp16_scaler = None
    if args.use_fp16:
        fp16_scaler = torch.cuda.amp.GradScaler()

    print(f"Loss, optimizer and schedulers ready.")

    # ============ optionally resume training ... ============
    to_restore = {"epoch": 0, 'best_score': -np.inf}
    is_save_best = False
    if os.path.isfile(args.resume):
        utils.restart_from_checkpoint(
            #os.path.join(args.output_dir, "checkpoint.pth"),
            args.resume,
            model=model,
            run_variables=to_restore,
            optimizer=optimizer,
            fp16_scaler=fp16_scaler,
            scheduler=scheduler,
            criterion=criterion
        )

    start_epoch = to_restore["epoch"]
    best_score = to_restore['best_score']

    # checkpoint storage
    from collections import namedtuple
    Checkpoint = namedtuple("Checkpoint", field_names=["obj", "logpath", "metric"])
    storage = []

    start_time = time.time()
    print("Starting training !")
    for epoch in range(start_epoch, args.epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)

        # ============ training one epoch ... ============
        train_stats = train_one_epoch(model, model_ema, criterion,
                                      train_loader, optimizer, scheduler, epoch, fp16_scaler, True, args)

        # Distributed bn
        if args.distributed:
            timm.utils.distribute_bn(
                model, torch.distributed.get_world_size(), True)

        # ============ validate one epoch ... ============
        if valid_loader is not None:
            valid_stats = train_one_epoch(model, None, criterion,
                                          valid_loader, optimizer, scheduler, epoch, fp16_scaler, False, args)
        else:
            valid_stats = train_stats

        if model_ema is not None:
            if args.distributed:
                timm.utils.distribute_bn(
                    model_ema, torch.distributed.get_world_size(), True)
            ema_valid_stats = train_one_epoch(model_ema.module, None, criterion,
                                              valid_loader, optimizer, scheduler, epoch, fp16_scaler, False, args)

            current_score = max(valid_stats['dice'], ema_valid_stats['dice'])
        else:
            current_score = valid_stats['dice']

        # if scheduler.__class__.__name__ == 'ReduceLROnPlateau':
        #     scheduler.step(current_score)
        # else:
        #     scheduler.step()

        if current_score > best_score:
            best_score = current_score
            is_save_best = True
        else:
            is_save_best = False

        valid_stats['best_score'] = best_score

        # ============ writing logs ... ============
        save_dict = {
            'model': model.state_dict(),
            # 'ema': model_ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'epoch': epoch + 1,
            'args': args,
            'criterion': criterion.state_dict(),
            'best_score': best_score
        }

        if model_ema is not None:
            save_dict['ema'] = model_ema.state_dict()

        if fp16_scaler is not None:
            save_dict['fp16_scaler'] = fp16_scaler.state_dict()
        utils.save_on_master(save_dict, os.path.join(
            args.output_dir, 'checkpoint.pth'))
        if is_save_best:
            utils.save_on_master(save_dict, os.path.join(
                args.output_dir, 'best.pth'))

        # Keep only top 5 checkpoints
        ckpt_path = os.path.join(args.output_dir, f'checkpoint{epoch:04}.pth')
        utils.save_on_master(save_dict, ckpt_path)
        storage.append(Checkpoint(obj=save_dict, logpath=ckpt_path, metric=current_score))
        storage = sorted(storage, key=lambda x: x.metric, reverse=True)
        if len(storage) > 5:
            last_item = storage.pop(-1)
            if os.path.isfile(last_item.logpath):
                try:
                    os.remove(last_item.logpath)
                except OSError:
                    pass

        log_train_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                           'epoch': epoch}
        log_valid_stats = {**{f'valid_{k}': v for k, v in valid_stats.items()},
                           'epoch': epoch}
        if not args.distributed or utils.is_main_process():
            with (Path(args.output_dir) / "log.txt").open("a") as f:
                f.write(json.dumps(log_train_stats) + "\n")
                f.write(json.dumps(log_valid_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


def train_one_epoch(
        model,
        model_ema,
        criterion,
        data_loader,
        optimizer,
        scheduler,
        epoch,
        fp16_scaler,
        is_train,
        args):

    metric_fn = Metric(multilabel=args.multilabel)
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

        if args.aux:
            targets = [
                batch['target'].cuda(non_blocking=True),
                batch['empty'].cuda(non_blocking=True)
            ]
        else:
            targets = batch['target'].cuda(non_blocking=True)

        with torch.cuda.amp.autocast(fp16_scaler is not None):
            if not is_train:
                with torch.no_grad():
                    logits = model(images)
            else:
                logits = model(images)

            logits = nn.functional.interpolate(
                input=logits,
                size=targets.shape[-2:],
                mode='bilinear',
                align_corners=False)

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

            if model_ema is not None:
                model_ema.update(model)

            scheduler.step()

        # logging
        torch.cuda.synchronize()
        metric_logger.update(loss=loss.item())
        # metric_logger.update(dice=metric_dict['dice'])
        for k, v in metric_dict.items():
            metric_logger.update(**{k: v})
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(wd=optimizer.param_groups[0]["weight_decay"])
    # gather the stats from all processes
    if args.distributed:
        metric_logger.synchronize_between_processes()
    print(f"[{prefix}] Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def predict(args):
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v))
          for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    # ============ preparing data ... ============
    if args.dataset == 'uw-gi':
        valid_loader = get_dataset(args, name='valid')
    else:
        return

    # ============ building Clusformer ... ============
    model, model_ema = get_model(args, False)
    model = model.eval()

    if os.path.isfile(args.resume):
        utils.restart_from_checkpoint(
            args.resume,
            model=model,
        )

    all_preds = []
    all_auxs = []

    from tqdm import tqdm
    for batch in tqdm(valid_loader, total=len(valid_loader)):
        # move images to gpu
        images = batch['image'].cuda(non_blocking=True)
        targets = batch['target'].cuda(non_blocking=True)

        with torch.no_grad():
            logits, auxs = model(images)
            logits = logits.argmax(1)
        all_preds.append(logits.detach().cpu())
        all_auxs.append(auxs.detach().cpu())

    all_preds = np.concatenate(all_preds, axis=0)
    all_auxs = np.concatenate(all_auxs, axis=0)
    save_dir = "/".join(args.resume.split("/")[:-1])
    print(save_dir)
    np.save(f"{save_dir}/preds.npy", all_preds)
    np.save(f"{save_dir}/auxs.npy", all_preds)


def predict_pseudo(args):
    print("git:\n  {}\n".format(utils.get_sha()))
    print("\n".join("%s: %s" % (k, str(v))
          for k, v in sorted(dict(vars(args)).items())))
    cudnn.benchmark = True

    args.distributed = False

    # ============ preparing data ... ============
    if args.dataset in ['uw-gi', 'chaos']:
        valid_loader = get_dataset(args, name='valid')
    else:
        return

    # ============ building Clusformer ... ============

    models = []
    for fold in range(5):
        model, model_ema = get_model(args, False)
        model = model.eval()

        resume_path = args.resume.format(fold)
        if os.path.isfile(resume_path):
            utils.restart_from_checkpoint(
                resume_path,
                model=model,
            )

        models.append(model)

    all_preds = []

    from tqdm import tqdm

    for batch in tqdm(valid_loader, total=len(valid_loader)):
        images = batch['image'].cuda(non_blocking=True)
        img_paths = batch['img_path']

        with torch.no_grad():
            logits = 0
            for model in models:
                logits += model(images).sigmoid()
            logits = logits / len(models)

        batch_size = images.shape[0]
        for i in range(batch_size):
            img_path = img_paths[i]
            pred = logits[i].detach().cpu().numpy()  # 512 x 512
            mask = img_path.replace('DICOM_anon', 'mask')
            save_dir = "/".join(mask.split("/")[:-1])
            os.makedirs(save_dir, exist_ok=True)
            np.save(mask, pred)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train 2.5D', parents=[get_args_parser()])
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    if args.pred:
        predict(args)
    elif args.pseudo:
        predict_pseudo(args)
    else:
        # Train phase one
        train(args)
