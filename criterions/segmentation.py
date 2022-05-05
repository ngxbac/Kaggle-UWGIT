import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp

JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
DiceLoss = smp.losses.DiceLoss(mode='multilabel')
BCELoss = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)


def poly1_focal_loss(logits, labels, epsilon=1.0, gamma=2.0):
    # p, pt, FL, and Poly1 have shape [batch, num of classes].
    p = logits.sigmoid()
    pt = labels * p + (1 - labels) * (1 - p)
    ce_loss = F.binary_cross_entropy_with_logits(
        logits, labels, reduction="none")
    FL = ce_loss * ((1 - pt) ** gamma)
    Poly1 = FL + epsilon * (1 - pt)**(gamma + 1)
    Poly1 = Poly1.mean()
    return Poly1


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
