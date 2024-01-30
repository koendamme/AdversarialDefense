import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


class AdversarialDataset(Dataset):
    def __init__(self, annotation_file, categories_file, img_dir, noise_dir, img_transform=None, noise_transform=None):
        self.img_dir = img_dir
        self.noise_dir = noise_dir
        annotations = pd.read_csv(annotation_file)
        self.categories = pd.read_csv(categories_file)
        self.images = annotations["ImageId"]
        self.labels = annotations["TrueLabel"]
        self.targets = annotations["TargetClass"]
        self.img_transform = img_transform
        self.noise_transform = noise_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx] + ".png")

        image = plt.imread(img_path)

        if self.noise_dir is not None:
            noise_path = os.path.join(self.noise_dir, self.images[idx] + ".npy")
            with open(noise_path, 'rb') as f:
                noise = np.load(f)
                noise = torch.Tensor(noise)

            if self.img_transform:
                image = self.img_transform(image)

            assert noise.shape == image.shape
        else:
            noise = None

        return image, noise, self.labels[idx] - 1, self.targets[idx] - 1, self.images[idx]