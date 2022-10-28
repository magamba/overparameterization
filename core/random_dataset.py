# -*- coding: utf-8 -*-
"""
Random dataset types:
    - uniform: each pixel sampled for U[mean - std, mean + std], with mean and std matching the 
               per-pixel statistics for the chosen dataset.
    - normal: each pixel sampled from N(mean, std), with mean and std matching the per-pixel
              statistics of the chosen dataset.
    - hypersphere: each pixel sampled with uniform probability from within the unit hypersphere.
    - jitter (use transforms): add strong jitter (random noise) to each image.
"""

from PIL import Image
from typing import Any, Callable, Optional, Tuple
import numpy as np
from torchvision.datasets import VisionDataset
from scipy.special import gammainc
from functools import partialmethod

def build_dataset(cls, *args, **kwargs):
    class DatasetClass(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)
    return DatasetClass

class RandomDataset(VisionDataset):
    
    def __init__(
        self,
        train_size : int,
        test_size: int,
        num_classes: int,
        data_shape : Tuple[int],
        train_mean: Tuple[float],
        train_std: Tuple[float],
        data_sample_seed: int,
        download: bool = None,
        train: bool = True,
        dist: str = "uniform",
        *args,
        **kwargs
    ) -> None:
    
        super(RandomDataset, self).__init__(*args, **kwargs)
        self.data_shape = data_shape
        self.data_sample_seed = data_sample_seed
        self.num_classes = num_classes
        self.test_size = test_size
        self.train = train
        self.train_size = train_size
        self.train_mean = np.asarray(train_mean)
        self.train_std = np.asarray(train_std)
        self.dist = dist
        
        if download:
            raise NotImplementedError("Random datasets are generated on the fly and cannot be downloaded.")
            
        self.data: Any = []
        self.targets = []
        
        # init PRNG
        rng = np.random.default_rng(data_sample_seed)
        nsamples = self.train_size if self.train else self.test_size
        
        # generate train data
        if self.dist == "uniform":
            self.data = rng.uniform(
                low=(self.train_mean - self.train_std), 
                high=(self.train_mean + self.train_std),
                size=(nsamples * np.prod(self.data_shape[1:]), self.data_shape[0])
            )
        elif self.dist == "normal":
            self.data = rng.normal(
                loc=self.train_mean,
                scale=self.train_std,
                size=(nsamples * np.prod(self.data_shape[1:]), self.data_shape[0])
            )        
        elif self.dist == "hypersphere":
            self.data = rng.standard_normal(
                size=(nsamples * np.prod(self.data_shape[1:]), self.data_shape[0])
            )
        else:
            raise ValueError("Unsupported distribution: {}".format(dist))
        
        self.data = np.vstack(
            tuple(self.data[:,i].reshape((-1,) + self.data_shape[1:]) for i in range(len(self.train_mean)))
        )
        self.data = self.data.reshape((self.data_shape[0], -1,) + self.data_shape[1:]).transpose(1,2,3,0) # HWC
        
        # generate labels
        self.targets = list(rng.integers(low=0, high=self.num_classes, size=nsamples, dtype=np.long))
    
        if self.dist == "hypersphere":
            # for reproducibility, after all data and targets are sampled, we normalize
            # hypersphere datasets so that each data point falls within the unit hypersphere
            #
            # we treat every channel separately, to match pixel standardization
            self.data = self.data.transpose(0,3,1,2).reshape((nsamples * self.data_shape[0], -1)) # NC,HW
            npoints = self.data.shape[0]
            ssq = np.sum(np.square(self.data), axis=1)
            ndims = np.prod(tuple(self.data_shape)[1:])
            
            # rescale points to fall within unit sphere with equal distribution
            # source: https://stackoverflow.com/a/44782884/14216894
            random_rescaling = gammainc(ndims / 2., ssq / 2.)**(1. / ndims) / np.sqrt(ssq)
            random_rescaling = np.tile(
                random_rescaling.reshape(npoints,1),
                (1, ndims)
            )
            self.data = np.multiply(self.data, random_rescaling).reshape((nsamples,) + self.data_shape).transpose(0,2,3,1) # HWC
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index
            
        Returns:
            tuple: (image, target) where target is index of the target class
        """
        img, target = self.data[index], self.targets[index]
        
        img = Image.fromarray((img * 255).astype(np.uint8))
        
        if self.transform is not None:
            img = self.transform(img)
            
        if self.target_transform is not None:
            target = self.target_transform(target)
            
        return img, target
        
    def __len__(self) -> int:
        return len(self.data)
    
    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


RANDOM_DATASETS_MAP = {
    "uniform_cifar10": build_dataset(
        RandomDataset, train_size=50000, test_size=10000, num_classes=10, data_shape=(3,32,32), train_mean=(0.4914, 0.4822, 0.4465), train_std=(0.2023, 0.1994, 0.2010), data_sample_seed=1234, dist="uniform"
    ),
    "normal_cifar10": build_dataset(
        RandomDataset, train_size=50000, test_size=10000, num_classes=10, data_shape=(3,32,32), train_mean=(0.4914, 0.4822, 0.4465), train_std=(0.2023, 0.1994, 0.2010), data_sample_seed=1234, dist="normal"
    ),
    "hypersphere_cifar10": build_dataset(
        RandomDataset, train_size=50000, test_size=10000, num_classes=10, data_shape=(3,32,32), train_mean=(0., 0., 0.), train_std=(1., 1., 1.), data_sample_seed=1234, dist="hypersphere"
    ),
    "uniform_cifar100": build_dataset(
        RandomDataset, train_size=50000, test_size=10000, num_classes=100, data_shape=(3,32,32), train_mean=(0.5071, 0.4865, 0.4409), train_std=(0.2009, 0.1984, 0.2023), data_sample_seed=1234, dist="uniform"
    ),
    "normal_cifar100": build_dataset(
        RandomDataset, train_size=50000, test_size=10000, num_classes=100, data_shape=(3,32,32), train_mean=(0.5071, 0.4865, 0.4409), train_std=(0.2009, 0.1984, 0.2023), data_sample_seed=1234, dist="normal"
    ),
    "hypersphere_cifar100": build_dataset(
        RandomDataset, train_size=50000, test_size=10000, num_classes=100, data_shape=(3,32,32), train_mean=(0., 0., 0.), train_std=(1., 1., 1.), data_sample_seed=1234, dist="hypersphere"
    ),
}

RANDOM_DATASETS = (
    "uniform_cifar10",
    "normal_cifar10",
    "hypersphere_cifar10",
    "uniform_cifar100",
    "normal_cifar100",
    "hypersphere_cifar100",
)

__random_datasets_all__ = [
    "uniform", "normal", "hypersphere", "jitter",
]
