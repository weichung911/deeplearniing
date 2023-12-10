import os
from PIL import Image
import torch

class TestImageFolder(torch.utils.data.Dataset):
    def __init__(self, root, transform=None):
        self.imgs = []
        for filename in os.listdir('./test/'):
            if filename.endswith('jpg'):
                self.imgs.append('{}'.format(filename))
        self.root = root
        self.transform = transform

    def __getitem__(self, index):
        filename = self.imgs[index]
        img = Image.open(os.path.join(self.root, filename))
        if self.transform is not None:
            img = self.transform(img)
            return img, filename

    def __len__(self):
        return len(self.imgs)