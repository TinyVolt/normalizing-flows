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

    def __len__(self):
        return len(self.array)

    def __getitem__(self, index):
        result = self.array[index]
        result = np.transpose(result, (2,0,1)) 
        result += np.random.uniform(low=0, high=0.25, size=result.shape)
        return result

train_loader = DataLoader(ShapesDataset(training_data), shuffle=True, batch_size=128)
test_loader = DataLoader(ShapesDataset(testing_data), shuffle=True, batch_size=128)
