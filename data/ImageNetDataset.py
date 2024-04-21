import tarfile
import torch
from torch.utils.data import Dataset, DataLoader

class ImageNetDataset(Dataset):
    def __init__(self, tar_path, transform=None):
        self.tar_path = tar_path
        self.transform = transform
        self.tar = tarfile.open(self.tar_path, 'r')
        self.member_names = self.tar.getnames()

    def __del__(self):
        self.tar.close()

    def __len__(self):
        return len(self.member_names)

    def __getitem__(self, idx):
        member = self.tar.getmember(self.member_names[idx])
        with self.tar.extractfile(member) as f:
            content = f.read()
        if self.transform:
            content = self.transform(content)
        return content



# Example usage:
if __name__ == "__main__":
    tar_path = "data/imagenet/val/ILSVRC2012_img_val.tar"  # Path to tarfile
    
    def my_transform(content):
        # Example: convert content to a PyTorch tensor
        return torch.tensor(content)

    dataset = ImageNetDataset(tar_path, transform=my_transform)
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch in dataloader:
        # Process each batch (e.g., train a model)
        print(f"Batch size: {len(batch)}")
