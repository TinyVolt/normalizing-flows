import torch
import torch.nn as nn
from torch.distributions.normal import Normal

class Flow1d(nn.Module):
    def __init__(self, n_components):
        super(Flow1d, self).__init__()
        self.mus = nn.Parameter(torch.randn(n_components), requires_grad=True)
        self.log_sigmas = nn.Parameter(torch.zeros(n_components), requires_grad=True)
        self.weight_logits = nn.Parameter(torch.ones(n_components), requires_grad=True)

    def forward(self, x):
        x = x.view(-1,1)
        weights = self.weight_logits.softmax(dim=0).view(1,-1)
        distribution = Normal(self.mus, self.log_sigmas.exp())
        z = (distribution.cdf(x) * weights).sum(dim=1)
        dz_by_dx = (distribution.log_prob(x).exp() * weights).sum(dim=1)
        return z, dz_by_dx


class CDFParams(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_hidden_layers=3, output_size=None):
        super(CDFParams, self).__init__()
        layers = [nn.Linear(input_size, hidden_size)]
        for _ in range(num_hidden_layers - 1):
            layers.append( nn.Linear(hidden_size, hidden_size) )
            layers.append( nn.ReLU() )
        layers.append( nn.Linear(hidden_size, output_size) )
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ConditionalFlow1D(nn.Module):
    def __init__(self, n_components):
        super(ConditionalFlow1D, self).__init__()
        self.cdf = CDFParams(output_size=n_components*3)

    def forward(self, x, condition):
        x = x.view(-1,1)
        mus, log_sigmas, weight_logits = torch.chunk(self.cdf(condition), 3, dim=1)
        weights = weight_logits.softmax(dim=1)
        distribution = Normal(mus, log_sigmas.exp())
        z = (distribution.cdf(x) * weights).sum(dim=1)
        dz_by_dx = (distribution.log_prob(x).exp() * weights).sum(dim=1)
        return z, dz_by_dx


class Flow2d(nn.Module):
    def __init__(self, n_components):
        super(Flow2d, self).__init__()
        self.flow_dim1 = Flow1d(n_components)
        self.flow_dim2 = ConditionalFlow1D(n_components)

    def forward(self, x):
        x1, x2 = torch.chunk(x, 2, dim=1)
        z1, dz1_by_dx1 = self.flow_dim1(x1)
        z2, dz2_by_dx2 = self.flow_dim2(x2, condition=x1)
        z = torch.cat([z1.unsqueeze(1), z2.unsqueeze(1)], dim=1)
        dz_by_dx = torch.cat([dz1_by_dx1.unsqueeze(1), dz2_by_dx2.unsqueeze(1)], dim=1)
        return z, dz_by_dx