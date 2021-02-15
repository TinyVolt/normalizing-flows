import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader

FILENAME = 'celeb.pkl'
with open(FILENAME, 'rb') as f:
    data = pickle.load(f)

# training and testing data are np arrays of shapes (20000, 32, 32, 3), (6838, 32, 32, 3)
training_data, testing_data = data['train'], data['test']
# training and testing data are np arrays of shapes (20000, 3, 32, 32), (6838, 3, 32, 32)
training_data = np.transpose(training_data, (0,3,1,2))
testing_data = np.transpose(testing_data, (0,3,1,2))
INPUT_H, INPUT_W = training_data.shape[2], training_data.shape[3]


class CelebDataset(Dataset):
    def __init__(self, array):
        self.array = array.astype(np.float32)

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return self.array[index]


train_loader = DataLoader(CelebDataset(training_data), shuffle=True, batch_size=128)
test_loader = DataLoader(CelebDataset(testing_data), shuffle=True, batch_size=128)