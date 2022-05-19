from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
# from monai.networks.nets import efficientnet
from models import efficientnet
from monai.networks.layers.factories import Conv, Act
from monai.networks.layers.utils import get_act_layer, get_norm_layer
from monai.networks.blocks.squeeze_and_excitation import SEBlock


class SegmentationHead(nn.Sequential):
    def __init__(self, spatial_dims, in_channels, out_channels, kernel_size=3):
        conv2d = Conv[Conv.CONV, spatial_dims](
            in_channels, out_channels,
            kernel_size=kernel_size, padding=kernel_size // 2
        )
        upsampling = nn.Identity()

        super().__init__(conv2d, upsampling)


class ConvReLU(nn.Sequential):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        out_channels,
        kernel_size,
        padding=0,
        stride=1,
        norm=("batch", {"eps": 1e-3, "momentum": 0.01})
    ):

        conv = Conv[Conv.CONV, spatial_dims](
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        relu = nn.ReLU()
        bn = get_norm_layer(
            name=norm, spatial_dims=spatial_dims, channels=out_channels)
        super(ConvReLU, self).__init__(conv, bn, relu)


class DecoderBlock(nn.Module):
    def __init__(
        self,
        spatial_dims,
        in_channels,
        skip_channels,
        out_channels,
        norm=("batch", {"eps": 1e-3, "momentum": 0.01})
    ):
        super().__init__()
        self.conv1 = ConvReLU(
            spatial_dims,
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm=norm,
        )
        # self.attention1 = SEBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=in_channels + skip_channels,
        #     n_chns_1=in_channels + skip_channels,
        #     n_chns_2=in_channels + skip_channels,
        #     n_chns_3=in_channels + skip_channels
        # )

        self.attention1 = nn.Identity()

        self.conv2 = ConvReLU(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm=norm,
        )
        self.attention2 = nn.Identity()
        # self.attention2 = SEBlock(
        #     spatial_dims=spatial_dims,
        #     in_channels=out_channels,
        #     n_chns_1=out_channels,
        #     n_chns_2=out_channels,
        #     n_chns_3=out_channels
        # )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x


class CenterBlock(nn.Sequential):
    def __init__(self, spatial_dims, in_channels, out_channels, norm=("batch", {"eps": 1e-3, "momentum": 0.01})):
        conv1 = ConvReLU(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm=norm
        )
        conv2 = ConvReLU(
            spatial_dims,
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            norm=norm
        )
        super().__init__(conv1, conv2)


class UnetDecoder(nn.Module):
    def __init__(
        self,
        spatial_dims,
        encoder_channels,
        decoder_channels,
        n_blocks=5,
        center=False,
        norm=("batch", {"eps": 1e-3, "momentum": 0.01})
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[1:]
        # reverse channels to start from head of encoder
        encoder_channels = encoder_channels[::-1]

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                spatial_dims,
                head_channels, head_channels, norm=norm)
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(norm=norm)
        blocks = [
            DecoderBlock(spatial_dims, in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        # remove first skip with same spatial resolution
        features = features[1:]
        # reverse channels to start from head of encoder
        features = features[::-1]

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)

        return x


class EfficientUnet(nn.Module):
    def __init__(self,
                 model_name,
                 output_channels=3,
                 spatial_dims=3,
                 pretrained=False,
                 in_channels=1,
                 encoder_depth=5,
                 decoder_channels=(256, 128, 64, 32, 16),
                 norm=("batch", {"eps": 1e-3, "momentum": 0.01})
                 ):
        super().__init__()

        if model_name == 'efficientnet-b0':
            out_channels = (in_channels, 32, 24, 40, 112, 320)
        elif model_name == 'efficientnet-b1':
            out_channels = (in_channels, 32, 24, 40, 112, 320)
        elif model_name == 'efficientnet-b2':
            out_channels = (in_channels, 32, 24, 48, 120, 352)
        elif model_name == 'efficientnet-b3':
            out_channels = (in_channels, 40, 32, 48, 136, 384)
        elif model_name == 'efficientnet-b4':
            out_channels = (in_channels, 48, 32, 56, 160, 448)
        elif model_name == 'efficientnet-b5':
            out_channels = (in_channels, 48, 40, 64, 176, 512)

        self.encoder = efficientnet.EfficientNetBNFeatures(
            model_name=model_name,
            pretrained=pretrained,
            spatial_dims=spatial_dims,
            in_channels=in_channels
        )

        self.decoder = UnetDecoder(
            spatial_dims=spatial_dims,
            encoder_channels=out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            norm=norm,
            center=False
        )

        self.segmentation_head = SegmentationHead(
            spatial_dims=spatial_dims,
            in_channels=decoder_channels[-1],
            out_channels=output_channels,
            kernel_size=3,
        )

    def forward(self, x):
        features = self.encoder(x)
        features = [features[0], features[2],
                    features[3], features[4], features[5]]
        features = [x] + features
        decoder_output = self.decoder(*features)

        masks = self.segmentation_head(decoder_output)

        return masks


if __name__ == '__main__':
    # import segmentation_models_pytorch as smp
    # model_smp = smp.Unet('timm-efficientnet-b3', in_channels=1, classes=3)
    # x = torch.rand((1, 1, 128, 128))
    # output = model_smp(x)
    # import pdb
    # pdb.set_trace()

    model = EfficientUnet(model_name='efficientnet-b3')
    x = torch.rand((1, 1, 96, 96, 64))
    output = model(x)
    print(output.shape)
