from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, params):
        self.params = params

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
