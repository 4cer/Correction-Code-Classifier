import os
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, filename):
        self.filename = filename

        # Get the path of the root directory
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Define the subfolder containing the datasets
        subfolder_name = 'dataset'

        # Construct the path to the file
        self.file_path = os.path.join(script_dir, subfolder_name, filename)

        with open(self.file_path, "r") as f:
            self.lines = f.readlines()

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        line = self.lines[index].strip()
        label = int(line.split(";")[0])
        text = line.split(";")[1]
        return (text, label)

# Example usage:
train_dataset = TextDataset("train.txt")


print(train_dataset.__getitem__(0))