import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset


class AdversarialDataset(Dataset):
    def __init__(self, annotation_file, categories_file, img_dir, img_extension,x_transform=None, y_transform=None):
        self.img_dir = img_dir
        self.img_extension = img_extension
        annotations = pd.read_csv(annotation_file)
        self.categories = pd.read_csv(categories_file)
        self.images = annotations["ImageId"]
        self.labels = annotations["TrueLabel"]
        self.targets = annotations["TargetClass"]
        self.x_transform = x_transform
        self.y_transform = y_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.images[idx] + self.img_extension)
        if self.img_extension == ".png":
            image = plt.imread(img_path)
        else:
            with open(img_path, 'rb') as f:
                image = np.load(f)
                image = np.transpose(image, (1, 2, 0))

        if self.x_transform:
            image = self.x_transform(image)

        return image, self.labels[idx] - 1, self.targets[idx] - 1, self.images[idx]