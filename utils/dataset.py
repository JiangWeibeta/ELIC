import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image


class ImageFolder(Dataset):
    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split
        self.split = split
        print(splitdir)
        if not splitdir.is_dir():
            raise RuntimeError(f'Invalid directory "{root}"')

        self.samples = [f for f in splitdir.iterdir() if f.is_file()]

        self.transform = transform

    def __getitem__(self, index):
        imgname = str(self.samples[index])
        img = np.array(Image.open(imgname).convert("RGB"))

        if self.transform:
            img = self.transform(img)
        return img

    def __len__(self):
        return len(self.samples)
