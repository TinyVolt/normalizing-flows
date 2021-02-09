import torch

from model import AutoRegressiveFlow
from data import train_loader, test_loader

def loss_function(target_distribution, z, log_dz_by_dx):
    log_likelihood = target_distribution.log_prob(z) + log_dz_by_dx
    return -log_likelihood.mean()

def train(model, train_loader, optimizer, target_distribution):
    model.train()
    for i, x in enumerate(train_loader):
        z, log_dz_by_dx = model(x)
        loss = loss_function(target_distribution, z, log_dz_by_dx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def eval_loss(model, data_loader, target_distribution):
    model.eval()
    total_loss = 0
    for x in data_loader:
        z, log_dz_by_dx = model(x)
        loss = loss_function(target_distribution, z, log_dz_by_dx)
        total_loss += loss * x.size(0)
    return (total_loss / len(data_loader.dataset)).item()

def train_and_eval(epochs, lr, train_loader, test_loader, target_distribution):
    flow = AutoRegressiveFlow(1, num_layers=5, n_components=10)
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        train(flow, train_loader, optimizer, target_distribution)
        train_losses.append(eval_loss(flow, train_loader, target_distribution))
        test_losses.append(eval_loss(flow, test_loader, target_distribution))
    return flow, train_losses, test_losses