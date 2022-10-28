# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader, random_split
from torchvision import datasets as torch_dsets, transforms as tx
from typing import Tuple, Union
from types import SimpleNamespace
import logging
from dataclasses import dataclass

from core.utils import DatasetInfo
from core.random_dataset import RANDOM_DATASETS_MAP, RANDOM_DATASETS

logger = logging.getLogger(__name__)

DATASETS = ("cifar10", "cifar100") + RANDOM_DATASETS
Datasets = SimpleNamespace(**{ds: ds for ds in DATASETS})

DATASET_INFO_MAP = {
    "cifar10": DatasetInfo("cifar10", (3, 32, 32), 10),
    "cifar100": DatasetInfo("cifar100", (3, 32, 32), 100),
}

DatasetInfos = SimpleNamespace(**DATASET_INFO_MAP)


def with_indices(datasetclass):
    """ Wraps a DataSet class, so that it returns (data, target, index, ground_truth).
    """
    def __getitem__(self, index):
        data, target = datasetclass.__getitem__(self, index)
        try:
            ground_truth = self._targets_orig[index]
        except AttributeError:
            ground_truth = target
        
        return data, target, index, ground_truth
        
    return type(datasetclass.__name__, (datasetclass,), {
        '__getitem__': __getitem__,
    })


""" Transforms
"""
class JitterTransform:
    """Adds random Jitter

    For a torchvision.datasets.vision.VisionDataset, we provide a transform that:
        adds random jitter with given probability

    @retun transformed_x: torch.Tensor transformed image
    """

    def __init__(self, probability=0.66, lower=0.02, upper=0.98, seed=1357):
        """Set up transformation parameters
        @param probability: per-pixel probability of sampling noise
        @param lower: smallest noise value to add to each perturbed pixel
        @param upper: largest noise value to add to each perturbed pixel
       
        """
        self.probability = probability
        self.lower = lower
        self.upper = upper
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def __call__(self, x):
        """
        @param x: torch.Tensor : image to be transformed
        @return torch.Tensor : transformed image
        """
        noise_mask = self.rng.uniform(low=0., high=1., size=tuple(x.shape))
        x[x >= self.probability] = self.upper
        x[x <= 1 - self.probability] = self.lower
        return x


""" Training transforms
"""
def _create_transforms(normalize, mean, std, **kwargs):
    crop_size = kwargs.pop("crop_size", 0)
    padding = kwargs.pop("padding", 4)
    hflip = kwargs.pop("hflip", False)
    jitter = kwargs.pop("jitter", False)

    transform_funcs = []
    if crop_size > 0:
        transform_funcs.append(tx.RandomCrop(crop_size, padding=padding))
    if hflip:
        transform_funcs.append(tx.RandomHorizontalFlip(p=0.5))

    transform_funcs.append(tx.ToTensor())
    if normalize:
        transform_funcs.append(tx.Normalize(mean, std))
    if jitter:
        transform_funcs.append(JitterTransform())
    return tx.Compose(transform_funcs)


""" Noisy labels
"""

def corrupt_labels(dset, noise_percent, seed=None):
    from numpy.random import default_rng
    rng = default_rng(seed)
    num_labels_to_corrupt = int(round(len(dset) * noise_percent))
    if num_labels_to_corrupt == 0:
        return
    if isinstance(dset, Subset):
        all_targets = dset.dataset.targets
        dset.dataset._targets_orig = dset.dataset.targets.copy()
        if isinstance(all_targets, list):
            all_targets = torch.tensor(all_targets)
        targets = all_targets[dset.indices]
    else:
        targets = dset.targets
        dset._targets_orig = dset.targets.copy()
        if isinstance(targets, list):
            targets = torch.tensor(targets)

    num_classes = targets.unique().shape[0]

    noise = torch.zeros_like(targets)
    if num_classes == 2:
        noise[0:num_labels_to_corrupt] = 1
    else:
        noise[0:num_labels_to_corrupt] = torch.from_numpy(
            rng.integers(1, num_classes, (num_labels_to_corrupt,))
        )
    shuffle = torch.from_numpy(rng.permutation(noise.shape[0]))
    noise = noise[shuffle]
    if isinstance(dset, Subset):
        all_noisy_targets = (targets + noise) % num_classes
        if isinstance(dset.dataset.targets, list):
            for idx, noisy_label in enumerate(all_noisy_targets.tolist()):
                dset.dataset.targets[dset.indices[idx]] = noisy_label
        else:
            dset.dataset.targets[dset.indices] = all_noisy_targets
    else:
        dset.targets = (targets + noise) % num_classes


""" Dataset interface
"""

def create_dataset(
    args,
    train=True,
    normalize=False,
    augment=False,
    subset_pct=None,
    validation=False,
    override_dset_class=None,
    **kwargs
):
    if args.data not in DATASETS:
        raise ValueError("{} is not a valid dataset".format(args.data))
        
    dset = None
    if args.random_dataset == "jitter":
        kwargs["jitter"] = True
    if args.data == Datasets.cifar10:
        if args.random_dataset is not None and args.random_dataset != "jitter":
            dclass = RANDOM_DATASETS_MAP[str(args.random_dataset) + "_" + str(args.data)]
            if args.random_dataset == "hypersphere":
                normalize = False
        else:
            dclass = torch_dsets.CIFAR10
    
        if override_dset_class is not None:
            CIFAR10 = override_dset_class(dclass)
        else:
            CIFAR10 = dclass
        
        gen_strategy = kwargs.pop("strategy", None)
        if train and augment:
            transforms = _create_transforms(
                normalize=normalize,
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
                crop_size=32,
                hflip=True,
                **kwargs
            )
        else:
            transforms = _create_transforms(
                normalize,
                (0.4914, 0.4822, 0.4465),
                (0.2023, 0.1994, 0.2010),
                **kwargs
            )
        dset = CIFAR10(
            root=args.data_dir, transform=transforms, train=train, download=False
        )
    elif args.data == Datasets.cifar100:
        if args.random_dataset is not None and args.random_dataset != "jitter":
            dclass = RANDOM_DATASETS_MAP[str(args.random_dataset) + "_" + str(args.data)]
            if args.random_dataset == "hypersphere":
                normalize = False
        else:
            dclass = torch_dsets.CIFAR100
    
        if override_dset_class is not None:
            CIFAR100 = override_dset_class(dclass)
        else:
            CIFAR100 = dclass
        gen_strategy = kwargs.pop("strategy", None)
        if train and augment:
            transforms = _create_transforms(
                normalize=normalize,
                mean=(0.5071, 0.4865, 0.4409),
                std=(0.2009, 0.1984, 0.2023),
                crop_size=32,
                hflip=True,
                **kwargs
            )
        else:
            transforms = _create_transforms(
                normalize,
                (0.5071, 0.4865, 0.4409),
                (0.2009, 0.1984, 0.2023),
                **kwargs
            )
        dset = CIFAR100(
            root=args.data_dir, transform=transforms, train=train, download=False
        )
    if subset_pct is not None and 1 > subset_pct > 0:
        rng = np.random.default_rng(args.data_split_seed)
        shuffle = rng.permutation(len(dset))
        split_index = int(subset_pct * len(dset))
        if validation:
            rand_indices = torch.from_numpy(shuffle)[split_index:]
        else:
            rand_indices = torch.from_numpy(shuffle)[:split_index]
        dset = Subset(dset, rand_indices)
    return dset


""" Data loaders
"""

@dataclass
class DataManager:
    dset: Dataset
    dloader: DataLoader
    tloader: DataLoader
    vloader: DataLoader
    vset: Dataset
    tset: Dataset


def create_data_manager(
    args,
    noise,
    seed=None,
    normalize=True,
    augment=False,
    train_validation_split=(None, None),
    train_subset_pct=None,
    test_subset_pct=None,
    override_dset_class=None,
    **kwargs
):
    dset = create_dataset(
        args,
        train=True,
        normalize=normalize,
        augment=augment,
        subset_pct=train_subset_pct,
        override_dset_class=override_dset_class,
        **kwargs
    )
    tset = create_dataset(
        args,
        train=False,
        normalize=normalize,
        augment=False,
        subset_pct=train_subset_pct,
        override_dset_class=override_dset_class,
        **kwargs
    )
    vset, vloader = None, None
    if train_validation_split != (None, None):
        vset = create_dataset(
            args,
            train=True,
            normalize=normalize,
            augment=False,
            subset_pct=train_subset_pct,
            validation=True,
            override_dset_class=override_dset_class,
            **kwargs
        )
        if seed is not None:
            torch.manual_seed(seed)
        rng_state = torch.get_rng_state()
        logger.info("Splitting training set into train: {}, val: {}.".format(
                train_validation_split[0], train_validation_split[1]
            )
        )
        _, vset = random_split(
            vset, train_validation_split, generator=torch.Generator("cpu").manual_seed(
                args.data_split_seed
            )
        )
        dset, _ = random_split(
            dset, train_validation_split, generator=torch.Generator("cpu").manual_seed(
                args.data_split_seed
            )
        )
        torch.set_rng_state(rng_state)
    corrupt_labels(dset, noise, args.label_seed)

    pin_memory=True
    try:
        shuffle = args.gen_strategy is None
    except AttributeError:
        shuffle = True
    
    logger.info("Running with {} cpu workers.".format(args.workers))
    kwargs_train = {
        "batch_size": args.batch_size,
        "num_workers" : args.workers,
        "shuffle": shuffle,
        "pin_memory": pin_memory,
    }
    kwargs_no_train = {
        "batch_size": args.batch_size,
        "num_workers" : args.workers,
        "shuffle": False,
        "pin_memory": pin_memory,
    }
    
    if train_validation_split != (None, None):
        vloader = DataLoader(vset, **kwargs_no_train)
    dloader = DataLoader(dset, **kwargs_train)
    tloader = DataLoader(tset, **kwargs_no_train)

    return DataManager(
        dset,
        dloader,
        tloader,
        vloader,
        vset,
        tset,
    )
