from torch.utils import data

class MyDataset(data.Dataset):
    """
    Wrapper class for the dataset. Used with the Pytorch DataLoader
    """
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return self.samples.shape[0]

    def __getitem__(self, index):
        return self.samples[index], self.labels[index]