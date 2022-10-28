# -*- coding: utf-8 -*-

""" Samplers handle data streams for Monte Carlo integration.
    Each sampler implements the basic Iterable interface, and returns
    batches of data points generated according to the dataloader
    passed during construction.
    
    There are three kind of samplers:
    - Base: return batches of shape (N, *)
    
    where N is the number of training samples
          * is any number of input data dimensions (e.g. C, H, W) for image data
    
    The BaseSampler class yields data samples from dataloader,
    without returning augmented samples.
    
    See core/data.py for data augmentation algorithms, and
    core/strategies.py for instantiating specific sampling
    strategies.
    
    Samplers support data parallelism in the batch dimension N.
"""

import abc
from enum import Enum
import logging
import torch.nn as nn
import torch

from core.data import create_data_manager, with_indices

logger = logging.getLogger(__name__)

"""Samplers
"""

class MCSampler(abc.ABC):
    """ Monte Carlo sampling Iterable class.
    """
    def __init__(self, dataloader, cmd_args, **kwargs):
        self._dataloader = dataloader
        self._data_iterator = None
        self._data_shape = tuple(next(iter(self._dataloader))[0].shape)
        self._init_args(cmd_args, **kwargs)
        self._log_sampling_info(cmd_args)
    
    def _init_args(self, cmd_args, **kwargs):
        self.device = "cuda:0" if torch.cuda.is_available() and cmd_args.device is not "cpu" else "cpu"
        self._mc_sample_seed = cmd_args.mc_sample_seed # seed for reproducible MC sampling
        self._num_samples = cmd_args.num_samples # number of training samples to compute the metric on
        self._skip = cmd_args.skip if cmd_args.skip is not None else 0 # skip the first _skip batches of the dataloader (useful for single-GPU data parallelism)
        self._batch_size = cmd_args.batch_size
        self._uid = int(self._skip // cmd_args.batch_size) if cmd_args.skip is not None else None # uid used to save results to a unique file
        
        avail_points = len(self._dataloader) * cmd_args.batch_size
        assert self._num_samples <= avail_points, "Error:" + \
            "requested {} samples, but dataset contains only {} points.".format(
                    self._num_samples,
                    avail_points
                    )
        assert (self._num_samples % cmd_args.batch_size == 0), "Error: BATCH_SIZE: {} must divide NUM_SAMPLES: {}".format(cmd_args.batch_size, self._num_samples)
        self._num_samples //= cmd_args.batch_size 
        
    def _log_sampling_info(self, cmd_args):
        pass
    
    def __iter__(self):
        """ Instantiate an iterator over self.dataloader with batches visited in
            reproducible order.
        """
        # seeding torch to ensure paths are visited always in the same order
        rng_state = torch.get_rng_state()
        torch.manual_seed(self._mc_sample_seed)
        
        iterator = iter(self._dataloader)
        torch.set_rng_state(rng_state)        
        
        logger.info("Initializing dataloader with seed {} to ensure reproducibility.".format(self._mc_sample_seed))
        
        batch_size = 1
        index = 0
        while index < self._skip:
            batch = next(iterator)
            batch_size = batch[0].shape[0]
            index += batch_size
        
        self._data_iterator = iterator
        self._batch_counter = 0
        return self
        
    def __next__(self):
        if self._data_iterator is None:
            raise StopIteration
        
        if self._batch_counter < self._num_samples:
            self._batch_counter += 1
            return(next(self._data_iterator))
        else:
            raise StopIteration

    def __getstate__(self):
        """Delete dataloader and iterator before serialization.
        """
        state = self.__dict__.copy()
        for key in ["dataloader", "iterator"]:
            try:
                del state[key]
            except KeyError:
                pass
        return state
        
    @property
    @abc.abstractmethod
    def name(self):
        """Return the name for logging.
        """

    @property
    def uid(self):
        return self._uid

    @property
    def data_shape(self):
        return self._data_shape


class BaseSampler(MCSampler):
    """ Base sampler without Monte Carlo integration.
        Used for computing statistics on samples from a dataset split,
        without data augmentations.
    """
    def _log_sampling_info(self, cmd_args):
        logger.info(
            "Running with {} batches of size {}.".format(
                self._num_samples,
                cmd_args.batch_size,
            )
        )
        
    def name(self):
        return Samplers.BASE_SAMPLER.value


""" Utils
"""
def load_data_manager(cmd_args, **kwargs):
    """ Create data manager according to command-line arguments.
    """
    tv_split = (None, None)
    if cmd_args.train_split and cmd_args.val_split:
        tv_split = (cmd_args.train_split, cmd_args.val_split)

    data_manager = create_data_manager(
        cmd_args,
        cmd_args.label_noise,
        seed=cmd_args.label_seed,
        train_validation_split=tv_split,
        normalize=True,
        **kwargs
    )
    return data_manager


def init_sampling_strategy(cmd_args, **kwargs):
    """ Create Monte Carlo sampling strategy based on data augmentations of controlled strength.
        Passing strategy=None disables Monte Carlo sampling and returns the standard dataset.
    """
    kwargs["override_dset_class"] = with_indices
    
    data_manager = load_data_manager(cmd_args, **kwargs)
    if cmd_args.dataset_split == "train":
        dataloader = data_manager.dloader
    elif cmd_args.dataset_split == "val":
        dataloader = data_manager.vloader
    elif cmd_args.dataset_split == "test":
        dataloader = data_manager.tloader
    else:
        raise ValueError("Unrecognized dataset split: {}".format(cmd_args.dataset_split))
    
    return dataloader


class Samplers(Enum):
    BASE_SAMPLER = "none"


SAMPLERS = {
    Samplers.BASE_SAMPLER.value: BaseSampler,
}


def create_sampler(sampler_type, cmd_args, **kwargs):
    dataloader = init_sampling_strategy(cmd_args, **kwargs)
    sampler_factory = SAMPLERS[sampler_type]
    return sampler_factory(dataloader, cmd_args=cmd_args, **kwargs)
