# -*- coding: utf-8 -*-

import abc
from enum import Enum
import logging
import torch.nn as nn
import torch
import numpy as np
from functools import partial
from core.metric_helpers import get_jacobian_fn, accuracy, spectral_norm, confidence

logger = logging.getLogger(__name__)

class Metric(nn.Module, abc.ABC):
    """Base Metric class.
    
       Each metric is implemented as a torch.nn.Module and
       the metric computation is done within the forward method.
       
       Irrespective of the sampler used, a metric should always 
       return batched results of shape (N,), where N is the number
       of base data points.
       
    """
    def __init__(self, model, data_shape, cmd_args, **kwargs):
        super(Metric, self).__init__()
        self.device = torch.device("cuda:0") if cmd_args.device != "cpu" and torch.cuda.is_available() else torch.device("cpu")
        self.model = model.to(self.device, non_blocking=True)
        self.model = self.model.eval()
        
        # batch shape attributes
        assert len(data_shape) > 3, "Error: expecting batches of shape (*, C, H, W), got: {}".format(tuple(data_shape))
        self._data_shape = data_shape
        self._batch_dims = data_shape[:-3]
        self._batch_size = np.prod(self._batch_dims)
        self._nbatch_dims = len(self._batch_dims)
        self._flatten_batch_dims = (-1,) + tuple(data_shape[self._nbatch_dims:])
        self._normalize_dims = self._batch_dims
        
        self._requires_targets = False
        self.integrate_fn = self._set_mc_integration_fn()
        self._init_args(cmd_args, **kwargs)

    def _init_args(self, cmd_args, **kwargs):
        """Used to pass command-line arguments to each metric.
        """
        pass
    
    def _set_mc_integration_fn(self):
        """Set up function used for Monte Carlo integration
           of the metric.
           
           This essentially ties a metric instantiation to a sampling
           strategy:
            - for BaseSamplers, no MC integration is performed
        """
        if self._nbatch_dims == 1:
            # base sampler
            return None        
        else:
            raise ValueError("Unsupported sampler batch shape: {}.".format(data_shape))
        
        return mc_integrate

    @property
    def requires_targets(self):
        """ Return True if computing the metric requires access to targets.
        """
        return self._requires_targets
        
    @property
    @abc.abstractmethod
    def name(self):
        """Return the metric name for logging.
        """
        
    @abc.abstractmethod
    def forward_impl_(self, x, target=None):
        """Compute metric here
        
           A metric should always return batched results of shape
           (N, *), where N is the batch size, and * is any arbitrary
           number of output dimensions.
        """
        
    def forward(self, x, target=None):
        """ Compute metric and integrate
        """
        result_ = self.forward_impl_(x, target)
        if self.integrate_fn is not None:
            # integrated values might be tuples, 
            # while result_ is always a singleton
            result = self.integrate_fn(
                result_.type(torch.float32), 
                x
            )
        else:
            result = result_
        return result

    def __getstate__(self):
        """Delete model before serialization.
        """
        state = self.__dict__.copy()
        for key in ["model"]:
            try:
                del state[key]
            except KeyError:
                pass
        return state

    def __repr__(self):
        """Return string representation of metric and model
        """
        return "{}:\n{}".format(self.name(), self.model)

        
class CrossEntropy(Metric):
    """Cross-entropy loss"""

    def __init__(self, model, data_shape, cmd_args, **kwargs):
        super(CrossEntropy, self).__init__(model, data_shape, cmd_args, **kwargs)
        self._requires_targets = True
        self.criterion = nn.CrossEntropyLoss(reduction='none').to(self.device)
    
    def name(self):
        return Metrics.CROSSENTROPY.value

    def forward_impl_(self, x, target):
        with torch.no_grad():
            out = self.model(
                x.view(self._flatten_batch_dims)
            )
            repeat_factor = out.shape[0] // target.shape[0]
            loss = self.criterion(out, target.repeat_interleave(repeat_factor)).view(self._batch_dims)
        return loss


class Accuracy(Metric):
    """top-1 0/1 loss"""

    def __init__(self, model, data_shape, cmd_args, **kwargs):
        super(Accuracy, self).__init__(model, data_shape, cmd_args, **kwargs)
        self._requires_targets = True

    def name(self):
        return Metrics.ACCURACY.value
        
    def forward_impl_(self, x, target):
        with torch.no_grad():
            out = self.model(
                x.view(self._flatten_batch_dims)
            )
            acc = accuracy(out, target).view(self._batch_dims).type(torch.int)
        return acc


class Confidence(Metric):
    """Confidence in predictions (distance from 1-hot encoding)"""

    def __init__(self, model, data_shape, cmd_args, **kwargs):
        super(Confidence, self).__init__(model, data_shape, cmd_args, **kwargs)
        self._requires_targets = True
        self.criterion = nn.Softmax(dim=-1).to(self.device)
        
    def _init_args(self, cmd_args, **kwargs):
        super(Confidence, self)._init_args(cmd_args, **kwargs)
        nclasses = kwargs.pop("nclasses", 0)
        assert nclasses > 0, "Missing required argument for {} metric: nclasses".format(self.name())
        self.confidence = partial(confidence, nclasses=nclasses)
    
    def name(self):
        return Metrics.CONFIDENCE.value

    def forward_impl_(self, x, target):
        with torch.no_grad():
            out = self.model(
                x.view(self._flatten_batch_dims)
            )
            conf = self.confidence(
                self.criterion(out),
                target
            ).view(self._batch_dims)
        return conf


class JacobianMetric(Metric):
    """Base class for evaluating Jacobian of a model w.r.t. to its input
    """
        
    def _init_args(self, cmd_args, **kwargs):
        super(JacobianMetric, self)._init_args(cmd_args, **kwargs)
        nclasses = kwargs.pop("nclasses", 0)
        assert nclasses > 0, "Missing required argument for {} metric: nclasses".format(self.name())
        self._target_logit_only = cmd_args.target_logit_only
        self._normalize_jacobian = cmd_args.normalization == "jacobian"
        self._requires_targets = cmd_args.target_logit_only or cmd_args.normalization == "crossentropy"
        self._jacobian_fn = get_jacobian_fn(
            self.model, self._batch_size, nclasses, cmd_args.bigmem, target_logit_only=self._target_logit_only, normalization=cmd_args.normalization,
        )


class JacobianNorm(JacobianMetric):
    """ Compute the Jacobian norm, optionally smoothed via MC integration
    """
    
    def name(self):
        return Metrics.JACOBIAN_NORM.value
    
    def forward_impl_(self, x, target=None):
        """ Compute the Jacobian norm of self.model w.r.t. @x
            optionally restricting computation to the specified @target output
            dimensions.
        """        
        if self._requires_targets:
            jacobian = self._jacobian_fn(
                x.view(self._flatten_batch_dims), target.to(self.device, non_blocking=False)
            )
        else:
            jacobian = self._jacobian_fn(x.view(self._flatten_batch_dims))
        
        jacobian_norm = torch.norm(
            jacobian.view(self._batch_dims + (-1,)),
            p=2,
            dim=self._nbatch_dims,
        )
        
        return jacobian_norm


class JacobianOperatorNorm(JacobianMetric):
    """ Compute the Jacobian operator norm, optionally smoothed via MC integration
    """
    
    def _init_args(self, cmd_args, **kwargs):
        super(JacobianOperatorNorm, self)._init_args(cmd_args, **kwargs)
        self._num_power_iters = cmd_args.num_power_iters
        logger.info("Estimating Jacobian operator norm using {} power iterations.".format(self._num_power_iters))

    def name(self):
        return Metrics.JACOBIAN_OPERATOR_NORM.value
    
    def forward_impl_(self, x, target=None):
        """ Compute the Jacobian operator norm of self.model w.r.t. @x
            optionally restricting computation to the specified @target output
            dimensions.
        """        
        if self._requires_targets:
            jacobian = self._jacobian_fn(
                x.view(self._flatten_batch_dims), target.to(self.device, non_blocking=False)
            )
        else:
            jacobian = self._jacobian_fn(x.view(self._flatten_batch_dims))
        
        
        _, operator_norm, _ = spectral_norm(
            jacobian,
            num_steps=self._num_power_iters
        )
        return operator_norm.view(self._batch_dims)


class Metrics(Enum):
    JACOBIAN_NORM = "jacobian"
    ACCURACY = "accuracy"
    CROSSENTROPY = "crossentropy"
    CONFIDENCE = "confidence"
    JACOBIAN_OPERATOR_NORM = "jacobian_operator_norm"


METRICS = {
    Metrics.JACOBIAN_NORM.value: JacobianNorm,
    Metrics.ACCURACY.value: Accuracy,
    Metrics.CROSSENTROPY.value: CrossEntropy,
    Metrics.CONFIDENCE.value: Confidence,
    Metrics.JACOBIAN_OPERATOR_NORM.value: JacobianOperatorNorm,
}


def create_metric(metric_name, model, data_shape, cmd_args=None, **kwargs):
    metric_factory = METRICS[metric_name]
    return metric_factory(model, data_shape, cmd_args=cmd_args, **kwargs)
