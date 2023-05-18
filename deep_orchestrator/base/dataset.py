from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, *args, **kwargs):
        self.params = kwargs

    def __len__(self):
        raise NotImplementedError("This method needs to be implemented in the child class")

    def __getitem__(self, idx):
        raise NotImplementedError("This method needs to be implemented in the child class")
