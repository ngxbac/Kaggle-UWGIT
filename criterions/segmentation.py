import torch
import torch.nn as nn
import torch.nn.functional as F
import segmentation_models_pytorch as smp

import warnings
from typing import Optional

import torch
import torch.nn as nn
from torch.nn.modules.loss import _Loss


def to_one_hot(labels: torch.Tensor, num_classes: int, dtype: torch.dtype = torch.float, dim: int = 1) -> torch.Tensor:
    # if `dim` is bigger, add singleton dim at the end
    if labels.ndim < dim + 1:
        shape = list(labels.shape) + [1] * (dim + 1 - len(labels.shape))
        labels = torch.reshape(labels, shape)

    sh = list(labels.shape)

    if sh[dim] != 1:
        raise AssertionError("labels should have a channel with length equal to one.")

    sh[dim] = num_classes

    o = torch.zeros(size=sh, dtype=dtype, device=labels.device)
    labels = o.scatter_(dim=dim, index=labels.long(), value=1)

    return labels


class PolyLoss(_Loss):
    def __init__(self,
                 softmax: bool = True,
                 ce_weight: Optional[torch.Tensor] = None,
                 reduction: str = 'mean',
                 epsilon: float = 1.0,
                 ) -> None:
        super().__init__()
        self.softmax = softmax
        self.reduction = reduction
        self.epsilon = epsilon
        self.cross_entropy = nn.CrossEntropyLoss(weight=ce_weight, reduction='none')

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: the shape should be BNH[WD], where N is the number of classes.
                You can pass logits or probabilities as input, if pass logit, must set softmax=True
            target: if target is in one-hot format, its shape should be BNH[WD],
                if it is not one-hot encoded, it should has shape B1H[WD] or BH[WD], where N is the number of classes,
                It should contain binary values
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
       """
        if len(input.shape) - len(target.shape) == 1:
            target = target.unsqueeze(1).long()
        n_pred_ch, n_target_ch = input.shape[1], target.shape[1]
        # target not in one-hot encode format, has shape B1H[WD]
        if n_pred_ch != n_target_ch:
            # squeeze out the channel dimension of size 1 to calculate ce loss
            self.ce_loss = self.cross_entropy(input, torch.squeeze(target, dim=1).long())
            # convert into one-hot format to calculate ce loss
            target = to_one_hot(target, num_classes=n_pred_ch)
        else:
            # # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
            self.ce_loss = self.cross_entropy(input, torch.argmax(target, dim=1))

        if self.softmax:
            if n_pred_ch == 1:
                warnings.warn("single channel prediction, `softmax=True` ignored.")
            else:
                input = torch.softmax(input, 1)

        pt = (input * target).sum(dim=1)  # BH[WD]
        poly_loss = self.ce_loss + self.epsilon * (1 - pt)

        if self.reduction == 'mean':
            polyl = torch.mean(poly_loss)  # the batch and channel average
        elif self.reduction == 'sum':
            polyl = torch.sum(poly_loss)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            # BH[WD]
            polyl = poly_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return (polyl)


class PolyBCELoss(_Loss):
    def __init__(self,
                 reduction: str = 'mean',
                 epsilon: float = 1.0,
                 ) -> None:
        super().__init__()
        self.reduction = reduction
        self.epsilon = epsilon
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            input: (∗), where * means any number of dimensions.
            target: same shape as the input
        Raises:
            ValueError: When ``self.reduction`` is not one of ["mean", "sum", "none"].
       """

            # # target is in the one-hot format, convert to BH[WD] format to calculate ce loss
        self.bce_loss = self.bce(input, target)
        pt = torch.sigmoid(input)
        pt = torch.where(target ==1,pt,1-pt)
        poly_loss = self.bce_loss + self.epsilon * (1 - pt)

        if self.reduction == 'mean':
            polyl = torch.mean(poly_loss)  # the batch and channel average
        elif self.reduction == 'sum':
            polyl = torch.sum(poly_loss)  # sum over the batch and channel dims
        elif self.reduction == 'none':
            # BH[WD]
            polyl = poly_loss
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')
        return (polyl)


def dice_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1, 0))
    return dice


def iou_coef(y_true, y_pred, thr=0.5, dim=(2, 3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred > thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1, 0))
    return iou


def criterion_2d(y_pred, y_true):
    return 0.25*BCELoss(y_pred, y_true) + 0.25*TverskyLoss(y_pred, y_true) + 0.5 * poly1_focal_loss(y_pred, y_true)


class ComboLoss(nn.Module):
    def __init__(self, multilabel=True, dice=True, tvk=False, poly=False):
        super(ComboLoss, self).__init__()
        self.dice = dice
        self.tvk = tvk
        self.poly = poly

        mode = 'multilabel' if multilabel else 'multiclass'
        self.dice_loss_fn = smp.losses.DiceLoss(mode=mode)
        self.tvk_loss_fn = smp.losses.TverskyLoss(mode=mode)
        self.poly_loss_fn = PolyBCELoss() if multilabel else PolyLoss()

    def forward(self, y_pred, y_true):
        loss = 0
        count = 0
        if self.dice:
            loss += self.dice_loss_fn(y_pred, y_true)
            count += 1

        if self.tvk:
            loss += self.tvk_loss_fn(y_pred, y_true)
            count += 1

        if self.poly:
            loss += self.poly_loss_fn(y_pred, y_true)
            count += 1

        return loss / count


# from torch.nn.modules.loss import _Loss
# from monai.utils import LossReduction
# import segmentation_models_pytorch as smp


# class DiceBceLoss(_Loss):
#     def __init__(
#         self,
#         w_dice = 0.5,
#         w_bce = 0.5,
#         finetune_lb = -1,
#         reduction = LossReduction.MEAN,
#     ):
#         super().__init__(reduction=LossReduction(reduction).value)
#         self.w_dice = w_dice
#         self.w_bce = w_bce
#         self.dice_loss = smp.losses.DiceLoss(mode='multilabel')
#         self.bce_loss = smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.01)

#     def forward(self, pred, label):
#         return self.dice_loss(
#             torch.softmax(pred, 1)[:, 1:], label) * self.w_dice + self.bce_loss(pred[:, 1:], label) * self.w_bce


