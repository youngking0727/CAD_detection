from torch.utils.data import Dataset, DataLoader
import random


class ECGDataset(Dataset):

    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __getitem__(self, idx):
        signal = self.X[idx]
        target = self.y[idx]
        return signal, target

    def __len__(self):
        return self.X.shape[0]
    
    def get_dataloader(self, num_workers=4, batch_size=16, shuffle=True):
        data_loader = DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,)
        return data_loader


# 编码一个句子，但是要把
class MultECGDataset(Dataset):
    def __init__(self, signal_data, channel_data, index):
        self.signal_data = signal_data
        self.channel_data = channel_data
        self.index = []
        for i in index:
            for channel_data in self.channel_data[i]['encoding']:
                self.index.append([i, channel_data])
        random.shuffle(self.index)

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        person_id_channel = self.index[index]
        person_id = person_id_channel[0]
        channel_data = person_id_channel[1]
        return {
            "person_id": person_id,
            "prompt1": self.channel_data[person_id]['prompt1'],
            "prompt2": self.channel_data[person_id]['prompt2'],
            "prompt3": self.channel_data[person_id]['prompt3'],
            "prompt4": self.channel_data[person_id]['prompt4'],
            "encoding": channel_data,
            "signal_data": self.signal_data[person_id]
        }


class ECGDataCollator(object):

    def __call__(self, instances):
        return instances