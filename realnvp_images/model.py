import torch
import torch.nn as nn
from torch.distributions.uniform import Uniform

class Preprocess(nn.Module):
    def __init__(self):
        super(Preprocess, self).__init__()

    def forward(self, x, reverse=False):
        device = x.device
        if reverse:
            return (x.sigmoid() - 0.05) / 0.9
        x += Uniform(0.0, 1.0).sample(x.size()).to(device)
        x = 0.05 + 0.9 * x / 4.0
        z = torch.log(x) - torch.log(1-x)
        log_det_jacobian = -x.log() - (1-x).log() + torch.tensor(0.4/9).log().to(device)
        return z, log_det_jacobian


class WeightNormConv2d(nn.Module):
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
        self.mask = self.create_mask(height, width, top_left_zero) # (1,1,height,width)
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
            return x
        else:
            x = x * log_scale.exp() + shift
            return x, log_scale


class AffineChannelwiseTransform(nn.Module):
    def __init__(self, top_half_as_input):
        super(AffineChannelwiseTransform, self).__init__()
        self.top_half_as_input = top_half_as_input
        self.scale_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.shift_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.net = ResNet(in_channel=6, out_channel=12)

    def forward(self, x, reverse=False):
        # x.size() is (batch_size, 12, H//2, W//2)
        # fixed, not_fixed have size (batch_size, 6, H//2, W//2)
        if self.top_half_as_input:
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
        
        if self.top_half_as_input:
            x_modified = torch.cat([fixed, not_fixed], dim=1)
            log_scale = torch.cat([log_scale, torch.zeros_like(log_scale)], dim=1)
        else:
            x_modified = torch.cat([not_fixed, fixed], dim=1)
            log_scale = torch.cat([torch.zeros_like(log_scale), log_scale], dim=1)
        
        if reverse:
            return x_modified
        return x_modified, log_scale


class BatchNorm(nn.Module):
    def __init__(self, input_size, momentum=0.9, eps=1e-5):
        super(BatchNorm, self).__init__()
        self.momentum = momentum
        self.eps = eps

        self.log_gamma = nn.Parameter(torch.zeros(input_size))
        self.beta = nn.Parameter(torch.zeros(input_size))

        self.register_buffer('running_mean', torch.zeros(input_size))
        self.register_buffer('running_var', torch.zeros(input_size))

    def forward(self, x, reverse=False):
        if self.training:
            batch_mean = x.mean([0,2,3])
            batch_var = x.var([0,2,3], unbiased=False)

            self.running_mean.mul_(self.momentum).add_( (1-self.momentum) * batch_mean )
            self.running_var.mul_(self.momentum).add_( (1-self.momentum) * batch_var )
        else:
            batch_mean = self.running_mean
            batch_var = self.running_var

        batch_mean = batch_mean.view(1,-1,1,1)
        batch_var = batch_var.view(1,-1,1,1)
        log_gamma = self.log_gamma.view(1,-1,1,1)
        beta = self.beta.view(1,-1,1,1)

        if reverse:
            x_normalized = (x - beta) * torch.exp(-log_gamma)
            z = x_normalized * torch.sqrt(batch_var + self.eps) + batch_mean
            return z
        else:
            x_normalized = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            z = x_normalized * log_gamma + beta
            log_det_jacobian = log_gamma - 0.5 * torch.log(batch_var + self.eps)
            return z, log_det_jacobian


class RealNVP(nn.Module):
    def __init__(self, height, width):
        super(RealNVP, self).__init__()
        self.preprocess = Preprocess()
        self.transforms_checkered_1 = nn.ModuleList([
            AffineCheckerboardTransform(height, width, False),
            BatchNorm(3),
            AffineCheckerboardTransform(height, width, True),
            BatchNorm(3),
            AffineCheckerboardTransform(height, width, False),
            BatchNorm(3),
            AffineCheckerboardTransform(height, width, True)
        ])

        self.transforms_channelwise = nn.ModuleList([
            AffineChannelwiseTransform(True),
            BatchNorm(12),
            AffineChannelwiseTransform(False),
            BatchNorm(12),
            AffineChannelwiseTransform(True),
        ])

        self.transforms_checkered_2 = nn.ModuleList([
            AffineCheckerboardTransform(height, width, False),
            BatchNorm(3),
            AffineCheckerboardTransform(height, width, True),
            BatchNorm(3),
            AffineCheckerboardTransform(height, width, False)
        ])

    def squeeze(self, x):
        '''converts a (batch_size,1,4,4) tensor into a (batch_size,4,2,2) tensor'''
        batch_size, num_channels, height, width = x.size()
        x = x.reshape(batch_size, num_channels, height//2, 2, width//2, 2)
        x = x.permute(0,1,3,5,2,4)
        x = x.reshape(batch_size, num_channels*4, height//2, width//2)
        return x

    def unsqueeze(self, x):
        '''converts a (batch_size,4,2,2) tensor into a (batch_size,1,4,4) tensor'''
        batch_size, num_channels, height, width = x.size()
        x = x.reshape(batch_size, num_channels//4, 2, 2, height, width)
        x = x.permute(0,1,4,2,5,3)
        x = x.reshape(batch_size, num_channels//4, height*2, width*2)
        return x

    def forward(self, x):
        z, log_det_jacobian_total = x, torch.zeros_like(x)
        
        z, log_det_jacobian = self.preprocess(z)
        log_det_jacobian_total += log_det_jacobian

        for transform in self.transforms_checkered_1:
            z, log_det_jacobian = transform(z)
            log_det_jacobian_total += log_det_jacobian

        z = self.squeeze(z)
        log_det_jacobian_total = self.squeeze(log_det_jacobian_total)

        for transform in self.transforms_channelwise:
            z, log_det_jacobian = transform(z)
            log_det_jacobian_total += log_det_jacobian

        z = self.unsqueeze(z)
        log_det_jacobian_total = self.unsqueeze(log_det_jacobian_total)

        for transform in self.transforms_checkered_2:
            z, log_det_jacobian = transform(z)
            log_det_jacobian_total += log_det_jacobian

        return z, log_det_jacobian_total

    def inverse(self, z):
        x = z
        for transform in self.transforms_checkered_2[::-1]:
            x = transform(x, reverse=True)
        
        x = self.squeeze(x)
        for transform in self.transforms_channelwise[::-1]:
            x = transform(x, reverse=True)

        x = self.unsqueeze(x)
        for transform in self.transforms_checkered_1[::-1]:
            x = transform(x, reverse=True)

        x = self.preprocess(x, reverse=True)

        return x


if __name__ == '__main__':
    import torch
    x = torch.randn(10,3,32,32)
    with torch.no_grad():
        y,z = RealNVP(32,32)(x)
        print(y.size(), z.size())
        z = RealNVP(32,32).inverse(x)
        print(z.size())