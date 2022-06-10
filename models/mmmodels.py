import torch
from mmcv.utils import config
from mmseg.models import build_segmentor


def get_mmseg_models(args):
    cfg = config.Config.fromfile(args.mmcfg)
    cfg.model.decode_head.num_classes = 4
    model = build_segmentor(cfg.model)

    checkpoint = torch.hub.load_state_dict_from_url(cfg.checkpoint_file)
    state_dict = model.state_dict()

    count = 0
    for k, v in checkpoint['state_dict'].items():
        if k in state_dict:
            state_dict[k] = v
            count += 1

    print(f"[+] Loaded {count} params")
    model.load_state_dict(state_dict)

    return model


if __name__ == '__main__':
    import torch

    cfg = config.Config.fromfile("mmconfigs/upernet_convnext_tiny.py")
    # cfg.model.decode_head.loss_decode = get_loss(None)
    cfg.model.decode_head.num_classes = 4

    x = torch.rand((2, 3, 512, 512))  # .cuda()
    y = torch.zeros(2, 1, 512, 512).type(torch.int64)  # .cuda()
    model = build_segmentor(cfg.model)  # .cuda()

    # loss = model(img=x)
    import pdb
    pdb.set_trace()
