import torch
import torch.nn as nn
import torch.nn.functional as F

class MaskedConv2d(nn.Conv2d):
    def __init__(self, include_base_point, *args, **kwargs):
        super(MaskedConv2d, self).__init__(*args, **kwargs)
        self.register_buffer('mask', torch.zeros_like(self.weight))
        self.create_mask(include_base_point)

    def forward(self, x):
        return F.conv2d(x, self.weight * self.mask, self.bias, self.stride, self.padding, self.dilation, self.groups)

    def create_mask(self, include_base_point):
        h_by_2 = self.kernel_size[0] // 2
        w_by_2 = self.kernel_size[1] // 2
        self.mask[:, :, :h_by_2] = 1
        self.mask[:, :, h_by_2, :w_by_2] = 1
        if include_base_point:
            self.mask[:, :, h_by_2, w_by_2] = 1


class LayerNorm(nn.LayerNorm):
    def __init__(self, *args, **kwargs):
        super(LayerNorm, self).__init__(*args, **kwargs)

    def forward(self, x):
        x = x.permute(0,2,3,1).contiguous()
        x = super().forward(x)
        x = x.permute(0,3,1,2).contiguous()
        return x


class AutoRegressiveFlow(nn.Module):
    def __init__(self, num_channels_input, num_layers=5, num_channels_intermediate=64, kernel_size=7, n_components=2, **kwargs):
        super(AutoRegressiveFlow, self).__init__()
        first_layer = MaskedConv2d(False, num_channels_input, num_channels_intermediate, kernel_size=kernel_size, padding=kernel_size//2, **kwargs)
        model = [first_layer]
        block = lambda: MaskedConv2d(True, num_channels_intermediate, num_channels_intermediate, kernel_size=kernel_size, padding=kernel_size//2, **kwargs)
        for _ in range(num_layers):
            model.append(LayerNorm(num_channels_intermediate))
            model.append(nn.ReLU())
            model.append(block())
        second_last_layer = MaskedConv2d(True, num_channels_intermediate, num_channels_intermediate, 1, **kwargs)
        last_layer = MaskedConv2d(True, num_channels_intermediate, n_components * 3 * num_channels_input, 1, **kwargs)
        model.append(second_last_layer)
        model.append(last_layer)

        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)