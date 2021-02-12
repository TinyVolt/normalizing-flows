import torch

from data import train_loader, test_loader
from model import RealNVP, AffineTransform2D

def loss_function(target_distribution, z, log_det_jacobian):
    log_likelihood = target_distribution.log_prob(z).sum(1) + log_det_jacobian.sum(1)
    return -log_likelihood.mean()

def train(model, train_loader, optimizer, target_distribution):
    model.train()
    for x in train_loader:
        x = x.float()
        z, dz_by_dx = model(x)
        loss = loss_function(target_distribution, z, dz_by_dx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def eval_loss(model, data_loader, target_distribution):
    model.eval()
    total_loss = 0
    for x in data_loader:
        x = x.float()
        z, dz_by_dx = model(x)
        loss = loss_function(target_distribution, z, dz_by_dx)
        total_loss += loss * x.size(0)
    return (total_loss / len(data_loader.dataset)).item()

def train_and_eval(epochs, lr, train_loader, test_loader, target_distribution):
    transforms = [AffineTransform2D(True), AffineTransform2D(False), AffineTransform2D(True), AffineTransform2D(False)]
    flow = RealNVP(transforms)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        train(flow, train_loader, optimizer, target_distribution)
        train_losses.append(eval_loss(flow, train_loader, target_distribution))
        test_losses.append(eval_loss(flow, test_loader, target_distribution))
    return flow, train_losses, test_losses