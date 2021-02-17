import torch

from data import train_loader, test_loader, INPUT_H, INPUT_W
from model import RealNVP

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def loss_function(target_distribution, z, log_det_jacobian):
    log_likelihood = target_distribution.log_prob(z) + log_det_jacobian
    return -log_likelihood.mean()

def train(model, train_loader, optimizer, target_distribution):
    model.train()
    for i, x in enumerate(train_loader):
        x = x.to(device)
        z, log_dz_by_dx = model(x)
        loss = loss_function(target_distribution, z, log_dz_by_dx)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print('Loss at iteration {} is {}'.format(i, loss.cpu().item()))

def eval_loss(model, data_loader, target_distribution):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x in data_loader:
            x = x.to(device)
            z, log_dz_by_dx = model(x)
            loss = loss_function(target_distribution, z, log_dz_by_dx)
            total_loss += loss * x.size(0)
    return (total_loss / len(data_loader.dataset)).cpu().item()

def train_and_eval(flow, epochs, lr, train_loader, test_loader, target_distribution):
    print('no of parameters is', sum(param.numel() for param in flow.parameters()))
    optimizer = torch.optim.Adam(flow.parameters(), lr=lr)
    train_losses, test_losses = [], []
    for epoch in range(epochs):
        print('Starting epoch:', epoch+1, 'of', epochs)
        train(flow, train_loader, optimizer, target_distribution)
        train_losses.append(eval_loss(flow, train_loader, target_distribution))
        test_losses.append(eval_loss(flow, test_loader, target_distribution))
    return flow, train_losses, test_losses

if __name__ == '__main__':
    print('Device is:', device)
    from torch.distributions.normal import Normal
    import numpy as np

    flow = RealNVP(INPUT_H, INPUT_W)
    target_distribution = Normal(torch.tensor(0).float().to(device),torch.tensor(1).float().to(device))
    flow, train_losses, test_losses = train_and_eval(flow, 1, 5e-4, train_loader, test_loader, target_distribution)
    print('train losses are', train_losses)
    print('test losses are', test_losses)
    torch.save(flow.state_dict(), 'trained_weights.pt')