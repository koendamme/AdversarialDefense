import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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

        noise = np.random.normal(0, 1, image.shape)
        # with open(img_path, 'rb') as f:
        #     noise = np.load(f)

        if self.img_transform:
            image = self.img_transform(image)

        if self.noise_transform:
            noise = self.noise_transform(noise)

        return image, noise, self.labels[idx] - 1, self.targets[idx] - 1, self.images[idx]