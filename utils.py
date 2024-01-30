import torch
import numpy as np

def correct_torch_image(image):
    permuted = torch.permute(image.cpu(), (1, 2, 0))
    return permuted


def minmax_scaler(arr, *, vmin=0, vmax=1):
    arr_min, arr_max = torch.min(arr), torch.max(arr)
    return ((arr - arr_min) / (arr_max - arr_min)) * (vmax - vmin) + vmin


def denormalize(img, means, stds):
    original_img = np.zeros(img.shape)

    for i in range(3):
        original_img[:, :, i] = img[:, :, i] * stds[i] + means[i]

    return original_img