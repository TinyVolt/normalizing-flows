import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal

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
        self.n_components = n_components

    def forward(self, x):
        batch_size, c_in = x.size(0), x.size(1) # x.size() is (B, c_in, h, w)
        h_and_w = x.size()[2:]
        out = self.model(x) # out.size() is (B, c_in * 3 * n_components, h, w)
        out = out.view(batch_size, 3 * self.n_components, c_in, *h_and_w) # out.size() is (B, 3*n_components, c_in, h, w)
        mus, log_sigmas, weight_logits = torch.chunk(out, 3, dim=1) # (B, n_components, c_in, h, w)

        # sizes are (B, n_components, c_in, h, w)
        # mus = mus.view(batch_size, self.n_components, c_in, *h_and_w)
        # log_sigmas = log_sigmas.view(batch_size, self.n_components, c_in, *h_and_w)
        # weight_logits = weight_logits.view(batch_size, self.n_components, c_in, *h_and_w)
        weights = F.softmax(weight_logits, dim=1)

        distribution = Normal(mus, log_sigmas.exp())

        x = x.unsqueeze(1) # x.size() is (B, 1, c_in, h, w)
        z = distribution.cdf(x) # z.size() is (B, n_components, c_in, h, w)
        z = (z * weights).sum(1) # z.size() is (B, c_in, h, w)

        log_dz_by_dx = (distribution.log_prob(x).exp() * weights).sum(1).log()

        return z, log_dz_by_dx
        
        
