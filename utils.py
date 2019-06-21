import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler


def load_split_train_test(datadir, valid_size=.2, batch_size=32, num_workers=6, size=(200, 200)):
    'Loads data into nice DataLoaders'
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train_transforms = transforms.Compose([transforms.Resize(size),
                                           transforms.ToTensor(),
                                           normalize,
                                       ])
    test_transforms = transforms.Compose([transforms.Resize(size),
                                          transforms.ToTensor(),
                                          normalize,
                                      ])
    train_data = datasets.ImageFolder(datadir,
                    transform=train_transforms)
    test_data = datasets.ImageFolder(datadir,
                    transform=test_transforms)
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)

    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    train = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=batch_size, num_workers=num_workers)
    test = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=batch_size, num_workers=num_workers)
    return train, test