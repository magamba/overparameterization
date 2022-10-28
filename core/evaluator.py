# -*- coding: utf-8 -*-

import logging
import torch

""" Evaluators take a metric, a sampler, and evaluate the metric on each 
    batch yielded by the sampler.
    
    Evaluators are independent from specific data streams,
    but metrics can assume a particular batch shape.
    
    However, a metric should always return batched results of shape (N, *),
    where N is the number of base data points.
    
    When defining new evaluators, a user should take care of matching a metric
    with a supported sampler.
    
    See core/metrics.py for available metrics, and core/sampling.py for
    samplers.
"""

logger = logging.getLogger(__name__)

class Evaluator:
    def __init__(self, metric, cmd_args, **kwargs):
        self.metric = metric
        self.results = {
            "stats" : [],
        }
        self._init_args(cmd_args, **kwargs)

    def _init_args(self, cmd_args, **kwargs):
        """Used to pass command-line arguments to the constructor.
        """
        self.device = torch.device("cuda:0") if cmd_args.device != "cpu" and torch.cuda.is_available() else torch.device("cpu")

    def __getstate__(self):
        """Delete sampler and metric before serialization.
        """
        state = self.__dict__.copy()
        for key in ["metric"]:
            try:
                del state[key]
            except KeyError:
                pass
        return state

    def __repr__(self):
        """ Return string representation of results
        """
        str_dict = {
            "metric": self.metric.name(),
            "results": self.results
        }
        return str(str_dict)

    def run(self, **kwargs) -> int:
        """Evaluate @self.metric on @self.sampler
        """
        stats = []
        logger.info("Computing metric {}".format(self.metric.name()))
        stat = self.metric(x=None, target=None).to("cpu", non_blocking=False)            
        stats += stat.tolist()
        self.results["stats"] = stats
        logger.info("Done with metric {}".format(self.metric.name()))
        
        return None


class DataEvaluator(Evaluator):
    def __init__(self, metric, sampler, cmd_args, **kwargs):
        super(DataEvaluator, self).__init__(metric, cmd_args, **kwargs)
        self.sampler = sampler
        self._batch_size = sampler.data_shape[0]
        self._requires_targets = metric.requires_targets
        self.results.update(
            {
                "sample_ids" : [],
                "targets" : [],
                "ground_truths" : [],
            }
        )
        self._init_args(cmd_args, **kwargs)

    def __getstate__(self):
        """Delete sampler and metric before serialization.
        """
        state = self.__dict__.copy()
        for key in ["sampler", "metric"]:
            try:
                del state[key]
            except KeyError:
                pass
        return state

    def __repr__(self):
        """ Return string representation of results
        """
        str_dict = {
            "metric": self.metric.name(),
            "sampler": self.sampler.name(),
            "results": self.results
        }
        return str(str_dict)

    def run(self, **kwargs) -> int:
        """Evaluate @self.metric on @self.sampler
        """
        sample_ids = []
        targets = []
        ground_truths = []
        stats = []
        uid = self.sampler.uid # unique id for data parallelism

        logger.info("Computing metric {}".format(self.metric.name()))

        for (x, target, sample_idx, ground_truth) in self.sampler:
            x = x.to(self.device, non_blocking=True)
            sample_ids += sample_idx.tolist()
            ground_truths += ground_truth.tolist()
            
            if self._requires_targets:
                target = target.to(self.device, non_blocking=True)
                stat = self.metric(x, target=target).view(
                    self._batch_size, -1
                )
                target = target.to("cpu", non_blocking=False)
            else:
                stat = self.metric(x).view(
                    self._batch_size, -1
                )
            
            stat = stat.to("cpu", non_blocking=False)
            stats += stat.tolist()
            targets += target.tolist()
        
        self.results["sample_ids"] = sample_ids
        self.results["targets"] = targets
        self.results["ground_truths"] = ground_truths
        self.results["stats"] = stats
        
        logger.info("Done with metric {}".format(self.metric.name()))
        
        return uid


def load_evaluator(metric, sampler, cmd_args, **kwargs):
    if sampler is None:
        return Evaluator(metric, cmd_args, **kwargs)
    else:
        return DataEvaluator(metric, sampler, cmd_args, **kwargs)
