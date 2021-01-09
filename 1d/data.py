import torch
import torch.utils.data as data 
import numpy as np

def generate_mixture_of_gaussians(num_of_points):
    n = num_of_points // 2
    gaussian1 = np.random.normal(loc=-1, scale=0.25, size=(n,))
    gaussian2 = np.random.normal(loc=0.5, scale=0.5, size=(num_of_points-n,))
    return np.concatenate([gaussian1, gaussian2])

class NumpyDataset(data.Dataset):
    def __init__(self, array):
        super().__init__()
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return self.array[index]

n_train, n_test = 2000, 1000
train_data = generate_mixture_of_gaussians(n_train)
test_data = generate_mixture_of_gaussians(n_test)

train_loader = data.DataLoader(NumpyDataset(train_data), batch_size=128, shuffle=True)
test_loader = data.DataLoader(NumpyDataset(test_data), batch_size=128, shuffle=True)