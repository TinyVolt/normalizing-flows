import torch
import torch.utils.data as data
from sklearn.datasets import make_moons


class NumpyDataset(data.Dataset):
    def __init__(self, array):
        super().__init__()
        self.array = array

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return self.array[index]


n_train, n_test = 2000, 1000
train_data, train_labels = make_moons(n_samples=n_train, noise=0.1)
test_data, test_labels = make_moons(n_samples=n_test, noise=0.1)

train_loader = data.DataLoader(NumpyDataset(train_data), batch_size=128, shuffle=True)
test_loader = data.DataLoader(NumpyDataset(test_data), batch_size=128, shuffle=True)