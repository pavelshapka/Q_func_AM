import os

import numpy as np
import torch
import torch.utils.data as data
from torchvision import transforms
from torchvision.datasets import CIFAR10

DATASET_PATH = "./datasets"
os.makedirs(DATASET_PATH, exist_ok=True)

train_dataset = CIFAR10(root=DATASET_PATH, train=True, download=True)
DATA_MEANS = (train_dataset.data / 255.0).mean(axis=(0,1,2))
DATA_STD = (train_dataset.data / 255.0).std(axis=(0,1,2))
del train_dataset


def image_to_numpy(img: torch.Tensor) -> np.ndarray:
    img = np.array(img, dtype=np.float32)
    img = (img - DATA_MEANS) / DATA_STD
    return img

def numpy_collate(batch) -> np.ndarray:
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple,list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)

def get_dataset() -> tuple[data.DataLoader, data.DataLoader, data.DataLoader]:
    test_augmentations = image_to_numpy
    train_augmentations = transforms.Compose([transforms.RandomHorizontalFlip(),
                                              transforms.RandomResizedCrop((32, 32), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                                              image_to_numpy])
    
    train_ds = CIFAR10(root=DATASET_PATH, train=True, transform=train_augmentations)
    val_ds = CIFAR10(root=DATASET_PATH, train=True, transform=test_augmentations)
    train_set, _ = torch.utils.data.random_split(train_ds, [45000, 5000], generator=torch.Generator().manual_seed(42))
    _, val_set = torch.utils.data.random_split(val_ds, [45000, 5000], generator=torch.Generator().manual_seed(42))

    test_set = CIFAR10(root=DATASET_PATH, train=False, transform=test_augmentations)

    train_loader = data.DataLoader(train_set,
                                   batch_size=128,shuffle=True,
                                   drop_last=True,
                                   collate_fn=numpy_collate,num_workers=8,
                                   persistent_workers=True)
    val_loader = data.DataLoader(val_set,
                                 batch_size=128,
                                 shuffle=False,
                                 drop_last=False,
                                 collate_fn=numpy_collate,
                                 num_workers=4,
                                 persistent_workers=True)
    test_loader = data.DataLoader(test_set,
                                  batch_size=128,
                                  shuffle=False,
                                  drop_last=False,
                                  collate_fn=numpy_collate,
                                  num_workers=4,
                                  persistent_workers=True)
    
    return train_loader, val_loader, test_loader
