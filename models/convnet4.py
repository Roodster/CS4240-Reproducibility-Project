import torch.nn as nn

from .models import register


def conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.MaxPool2d(2)
    )

# Conv block with adaptive avg pool
# As suggested in "Enhancing Few-Shot Learning in Lightweight Models via Dual-Faceted Knowledge Distillation"
def avg_pool_conv_block(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d((1,1))
    )


@register('convnet4')
class ConvNet4(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.out_dim = 1600

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.shape[0], -1)

# ConvNet4-64 (with avg pooling at the end)
# As proposed in "Enhancing Few-Shot Learning in Lightweight Models via Dual-Faceted Knowledge Distillation"
@register('convnet4-64')
class ConvNet4(nn.Module):

    def __init__(self, x_dim=3):
        super().__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, 64),
            conv_block(64, 64),
            conv_block(64, 64),
            avg_pool_conv_block(64, 64),
        )
        self.out_dim = 64

    def forward(self, x):
        x = self.encoder(x)
        return x.view(x.shape[0], -1)
