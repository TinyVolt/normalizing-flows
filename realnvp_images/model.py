import torch
import torch.nn as nn

class WeightNormConv2d(nn.Conv2d):
    def __init__(self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True):
        super(WeightNormConv2d, self).__init__()
        self.conv = nn.utils.weight_norm(nn.Conv2d(in_channel, out_channel, kernel_size, stride=stride, padding=padding, bias=bias))

    def forward(self, x):
        return self.conv(x)


class ResNetBlock(nn.Module):
    def __init__(self, dim):
        super(ResNetBlock, self).__init__()
        self.block = nn.Sequential(
            WeightNormConv2d(dim, dim, 1, stride=1, padding=0),
            nn.ReLU(),
            WeightNormConv2d(dim, dim, 3, stride=1, padding=1),
            nn.ReLU(),
            WeightNormConv2d(dim, dim, 1, stride=1, padding=0),
        )

    def forward(self, x):
        return x + self.block(x)


class ResNet(nn.Module):
    def __init__(self, in_channel=3, out_channel=6, intermediate_channel=128, num_blocks=8):
        super(ResNet, self).__init__()
        layers = [WeightNormConv2d(in_channel, intermediate_channel, 3, stride=1, padding=1), nn.ReLU()]
        for _ in range(num_blocks):
            layers.append( ResNetBlock(intermediate_channel) )
        layers.append(nn.ReLU())
        layers.append( WeightNormConv2d(intermediate_channel, out_channel, 3, stride=1, padding=1) )

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class AffineCheckerboardTransform(nn.Module):
    def __init__(self, height, width, top_left_zero=False):
        super(AffineCheckerboardTransform, self).__init__()
        self.mask = create_mask(top_left_zero) # (1,1,height,width)
        self.scale_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.shift_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.net = ResNet()

    def create_mask(self, height, width, top_left_zero):
        mask = (torch.arange(height).view(-1,1) + torch.arange(width)) 
        if not top_left_zero:
            mask += 1
        return (mask % 2).unsqueeze(0).unsqueeze(0)

    def forward(self, x, reverse=False):
        # x has size (batch_size, 3, height, width)
        x_masked = x * self.mask
        # log_scale and shift have size (batch_size, 3, height, width)
        log_scale, shift = self.net(x_masked).chunk(2, dim=1)
        log_scale = log_scale.tanh() * self.scale_scale + self.shift_scale

        log_scale = log_scale * (1 - self.mask)
        shift = shift * (1 - self.mask)
        if reverse:
            x = (x - shift) * torch.exp(-log_scale)
        else:
            x = x * log_scale.exp() + shift
        return x, log_scale

class AffineChannelwiseTransform(nn.Module):
    def __init__(self):
        super(AffineChannelwiseTransform, self).__init__()
        self.scale_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.shift_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.net = ResNet(in_channel=6, out_channel=12)

    def forward(self, x, top_half_as_input, reverse=False):
        # x.size() is (batch_size, 12, H//2, W//2)
        # fixed, not_fixed have size (batch_size, 6, H//2, W//2)
        if top_half_as_input:
            fixed, not_fixed = x.chunk(2, dim=1)
        else:
            not_fixed, fixed = x.chunk(2, dim=1)
        # log_scale and shift have size (batch_size, 6, H//2, W//2)
        log_scale, shift = self.net(fixed).chunk(2, dim=1)
        log_scale = log_scale.tanh() * self.scale_scale + self.shift_scale

        if reverse:
            not_fixed = (not_fixed - shift) * torch.exp(-log_scale)
        else:
            not_fixed = not_fixed * log_scale.exp() + shift
        
        if top_half_as_input:
            x_modified = torch.cat([fixed, not_fixed], dim=1)
            log_scale = torch.cat([log_scale, torch.zeros_like(log_scale)], dim=1)
        else:
            x_modified = torch.cat([not_fixed, fixed], dim=1)
            log_scale = torch.cat([torch.zeros_like(log_scale), log_scale], dim=1)
        return x_modified, log_scale


class RealNVP(nn.Module):
    def __init__(self):
        super(RealNVP, self).__init__()
        pass

    def forward(self, x):
        pass