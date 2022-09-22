from torch.utils.data import Dataset, DataLoader



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



class Single_Label_Dataloader(Dataset):
    