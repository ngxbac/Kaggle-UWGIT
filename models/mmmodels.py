from mmcv.utils import config
from mmseg.models import build_segmentor


def get_mmseg_models(args):
    cfg = config.Config.fromfile(args.mmcfg)
    # cfg.model.decode_head.loss_decode = get_loss(None)
    cfg.model.decode_head.num_classes = 4
    model = build_segmentor(cfg.model)
    return model


if __name__ == '__main__':
    import torch

    cfg = config.Config.fromfile("mmconfigs/fpn_r50.py")
    # cfg.model.decode_head.loss_decode = get_loss(None)
    cfg.model.decode_head.num_classes = 4

    x = torch.rand((2, 3, 512, 512)).cuda()
    y = torch.zeros(2, 1, 512, 512).type(torch.int64).cuda()
    model = build_segmentor(cfg.model).cuda()

    loss = model(img=x)
    import pdb
    pdb.set_trace()
