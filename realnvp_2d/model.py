import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, num_hidden_layers, output_size):
        super(MLP, self).__init__()
        layers = [nn.Linear(input_size, hidden_size)]
        for _ in range(num_hidden_layers - 1):
            layers.append( nn.Linear(hidden_size, hidden_size) )
            layers.append( nn.ReLU() )
        layers.append( nn.Linear(hidden_size, output_size) )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class AffineTransform2D(nn.Module):
    def __init__(self, left, hidden_size=256, num_hidden_layers=2):
        super(AffineTransform2D, self).__init__()
        self.mlp = MLP(2, hidden_size, num_hidden_layers, 2)
        self.mask = torch.FloatTensor([1,0]) if left else torch.FloatTensor([0,1])
        self.mask = self.mask.view(1,-1)
        self.scale_scale = nn.Parameter(torch.zeros(1), requires_grad=True)
        self.shift_scale = nn.Parameter(torch.zeros(1), requires_grad=True)

    def forward(self, x, reverse=False):
        # x.size() is (B,2)
        x_masked = x * self.mask
        # log_scale and shift have size (B,1)
        log_scale, shift = self.mlp(x_masked).chunk(2, dim=1)
        log_scale = log_scale.tanh() * self.scale_scale + self.shift_scale
        # log_scale and shift have size (B,2)
        shift = shift  * (1-self.mask)
        log_scale = log_scale * (1-self.mask)
        if reverse:
            x = (x - shift) * torch.exp(-log_scale)
        else:
            x = x * torch.exp(log_scale) + shift
        return x, log_scale


class RealNVP(nn.Module):
    def __init__(self, affine_transforms):
        super(RealNVP, self).__init__()
        self.transforms = nn.ModuleList(affine_transforms)

    def forward(self, x):
        z, log_det_jacobian = x, torch.zeros_like(x)
        for transform in self.transforms:
            z, log_scale = transform(z)
            log_det_jacobian += log_scale
        return z, log_det_jacobian

    def invert(self, z):
        for transform in self.transforms[::-1]:
            z, _ = self.transform(z)
        return z