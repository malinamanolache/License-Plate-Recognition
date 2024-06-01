from torch.utils.data import Dataset

class RodosolDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, root_dir: str, split: str):

        if split not in ["training", "validation", "testing"]:
            raise ValueError("`subset` must be one of ['training', 'validation', 'testing']")
        
        self.root_dir = root_dir
        self.split = split

    def _get_split_paths(self): list:
        filenames = []

        split_path = os.path.join(self.root_dir, "split.txt")
        with open(split_path, 'r') as file:
        for line in file:
            file_path, label = line.strip().split(';')

            if label == split:
                absolute_path =  os.path.normpath(os.path.join(self.root_dir, file_path))
                filenames.append(absolute_path)

        return filenames

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, idx):
        raise NotImplementedError