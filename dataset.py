import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms


class FaceDataset(Dataset):
    def __init__(self, data_path, image_size):
        self.data_path = data_path
        self.image_size = image_size

        self.images = [
            os.path.join(data_path, f)
            for f in os.listdir(data_path)
            if f.endswith(('.jpg', '.png', '.jpeg'))
        ]

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert('RGB')
        return self.transform(image)


def get_dataloader(data_path, config):
    dataset = FaceDataset(data_path, config['image_size'])
    return DataLoader(
        dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )