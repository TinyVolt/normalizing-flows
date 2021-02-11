import numpy as np
import pickle
from torch.utils.data import Dataset, DataLoader

FILENAME = 'shapes.pkl'
with open(FILENAME, 'rb') as f:
    data = pickle.load(f)
training_data, testing_data = data['train'], data['test']
# training_data.shape = (10479, 20, 20, 1)
training_data = (training_data > 127.5).astype(np.uint8)
# training_data.shape = (4491, 20, 20, 1)
testing_data = (testing_data > 127.5).astype(np.uint8)


class ShapesDataset(Dataset):
    def __init__(self, array):
        self.array = array.astype(np.float32) / 2.0
        self.array = np.transpose(self.array, (0,3,1,2))

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        return self.array[index]

train_loader = DataLoader(ShapesDataset(training_data), shuffle=True, batch_size=128)
test_loader = DataLoader(ShapesDataset(testing_data), shuffle=True, batch_size=128)
