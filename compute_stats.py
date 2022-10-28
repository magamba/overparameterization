# -*- coding: utf-8 -*-

"""
Compute metrics on a trained model loaded from one or more checkpoints.
"""

import torch

import sys
import os
from zipfile import ZipFile
from functools import partial
import logging
import json

from core.cmd import create_default_args
from core.data import DATASET_INFO_MAP
from core.utils import (
    init_torch,
    all_log_dir,
    checkpoint_dir,
    init_logging,
    init_prngs,
    prepare_dirs
)
from core.strategies import load_sampler, GEN_STRATEGIES
from models import factory
from core.metrics import METRICS, Metrics, create_metric
from core.evaluator import load_evaluator

logger = logging.getLogger(__name__)


def checkpointed_models(cmd_args, reverse=False, index_by_ordinal=True, device="cpu", init_only=False):
    checkpoints_loc = "{}.zip".format(checkpoint_dir(cmd_args))

    def step_key(zip_info):
        return int(zip_info.filename.split("_")[1].split(".")[0])

    with ZipFile(checkpoints_loc) as checkpoints_zip:
        zip_infos = [zi for zi in checkpoints_zip.infolist() if zi.file_size > 0]
        for cur_key, zip_info in enumerate(sorted(zip_infos, key=step_key, reverse=reverse)):
            if not index_by_ordinal:
                cur_key = step_key(zip_info)
                if cmd_args.nsteps_per_epoch is not None:
                    cur_key = int(cur_key // cmd_args.nsteps_per_epoch)
            if init_only:
                if int(cur_key) > 0:
                    continue
            elif (
                cmd_args.checkpoint is not None
                and cur_key not in cmd_args.checkpoint
            ):
                continue
            logger.info("Loading model checkpoint {}{}".format(
                "ordinal " if index_by_ordinal else "", cur_key
            ))
            with checkpoints_zip.open(zip_info.filename) as saved_file:
                saved_obj = torch.load(
                    saved_file, map_location=torch.device(device)
                )
                use_bn = True if 'batch_norm' in cmd_args.model_additions else False
                net = factory.create_model(
                    cmd_args.model, cmd_args.data, additions=cmd_args.model_additions, dropout_rate=cmd_args.dropout, use_batch_norm=use_bn
                )
                net.load_state_dict(saved_obj["model"])
                net = net.to(device=device, dtype=torch.get_default_dtype())
                net.eval()
                yield cur_key, net


def load_metric(args, model, sampler, device):
    """ Initialize metric based on args
    """
    data_shape = sampler.data_shape if sampler is not None else (1,3,32,32) # dummy shape
    kwargs = {}
    if args.metric in [ Metrics.JACOBIAN_NORM.value, Metrics.JACOBIAN_OPERATOR_NORM.value, Metrics.CONFIDENCE.value ]:
        kwargs["nclasses"] = DATASET_INFO_MAP[args.data].output_dimension
    
    logger.info("Loading metric: {}".format(args.metric))
    return create_metric(args.metric, model, data_shape, args, **kwargs)


def compute_stats(evaluator, step):
    logger.info("Evaluating checkpoint {}".format(step))
    uid = evaluator.run()
    logger.info("Done with step {}".format(step))
    return evaluator, uid


def add_local_args(parser):
    opt_group = parser.add_argument_group("compute_stats local")
    opt_group.add_argument(
        "--gen-strategy",
        default='none',
        choices=list(GEN_STRATEGIES.keys()),
    )
    
    opt_group.add_argument(
        "--metric",
        default=None,
        choices=list(METRICS.keys()),
        help="Metric to compute.",
    )
    
    opt_group.add_argument(
        "--normalization",
        default=None,
        type=str,
        choices=("jacobian", "softmax", "logsoftmax", "crossentropy"),
        help="Specify how the measure should be normalized. If not set, the network output and Jacobian will not be normalized.",
    )
    
    opt_group.add_argument(
        "--outname",
        type=str,
        default="",
        help="Base filename for saving results.",
    )
    
    opt_group.add_argument(
        "--dataset-split",
        type=str,
        default="train",
        help="The dataset split to use [default = train].",
    )
    
    opt_group.add_argument(
        "--bigmem",
        action='store_true',
        default=False,
        help="If enabled, compute full-dimensional Jacobian using a single backward pass. This is faster, but scales in the number of output dimensions of the network.",
    )
    
    opt_group.add_argument(
        "--target-logit-only",
        action='store_true',
        default=False,
        help="If enabled, compute Jacobian only of the output dimension of the target logit used for training. If NORMALIZATION is set to 'crossentropy', this option is ignored.",
    )
    
    opt_group.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number dataset samples to compute the measure on. It should be divisible by BATCH_SIZE, and its maximum value is the size of the dataset split considered.",
    )
    
    opt_group.add_argument(
        "--checkpoint",
        nargs="*",
        default=None,
        type=int,
        help="Checkpoint IDs to load. Leave blank to print all available checkpoints of the specified model.",
    )
    
    opt_group.add_argument(
        "--checkpoint-by-step",
        action='store_true',
        default=False,
        help="Index checkpoints by their (absolute) step number, rather than by ordinal id.",
    )

    opt_group.add_argument(
        "--nsteps-per-epoch",
        default=None,
        type=int,
        help="If CHECKPOINT_BY_STEP is enabled, it is possible to specify CHECKPOINTs to load in epochs rather than steps, by providing how many steps form an epoch.",
    )

    opt_group.add_argument(
        "--skip",
        default=None,
        type=int,
        help="Skip the first SKIP dataset samples [default = 0]. Useful for data parallelism."
    )
    
    opt_group.add_argument(
        "--mc-sample-seed",
        default=4321,
        type=int,
        help="Seed used for sampling batches from the dataloader."
    )
    
    opt_group.add_argument(
        "--num-power-iters",
        type=int,
        default=10,
        help="Number of power iterations used to estimate the spectral norm of the Jacobian. Used only for METRIC = jacobian_operator_norm.",
    )
    

def main(args, logs_path):
    logger.info(args)

    init_torch(cmd_args=args, double_precision=False)
    init_prngs(args)

    if torch.cuda.is_available() and args.device != "cpu":
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    if args.checkpoint is None:
        all_available_checkpoints = [
            step_key for (step_key, _) in checkpointed_models(args, reverse=False, index_by_ordinal=False)
        ]
        logger.info("Available checkpoints in {} are:".format(all_log_dir(args)))
        logger.info(all_available_checkpoints)
        sys.exit(0)
    
    sampler = load_sampler(args)
    checkpointed_models_gen = checkpointed_models(args, reverse=False, index_by_ordinal= not args.checkpoint_by_step, device=device)
    for step, model in checkpointed_models_gen:

        metric = load_metric(args, model, sampler, device)
        evaluator = load_evaluator(metric, sampler, args)
        
        evaluator, uid = compute_stats(evaluator, step)    
        results = evaluator.results
        
        metadata = {
            "metadata" : {
                "model" : args.model,
                "dataset" : args.data,
                "split" : args.dataset_split,
                "seed" : args.seed,
                "label-noise" : args.label_noise,
                "augmentation" : args.augmentation,
                "checkpoint" : step,
                "nsamples" : args.num_samples,
                "label-noise-seed" : args.label_seed,
                "data-split-seed" : args.data_split_seed,
                "mc-sample-seed" : args.mc_sample_seed,
                "metric" : metric.name(),
                "normalization": args.normalization,
                "sampling-strategy" : args.gen_strategy,
            }
        }
        
        results.update(metadata)
        
        outname = args.outname + '-' + args.metric if args.outname != '' else args.metric
        fname = os.path.join(logs_path, outname)
        fname += '-target_only' if args.target_logit_only and args.normalization != 'crossentropy' else ''
        fname += '-random_' + args.random_dataset if args.random_dataset is not None else ''
        if args.dataset_split is not None:
            fname += '-' + str(args.dataset_split)
        
        if args.gen_strategy is not None and args.gen_strategy != 'none':
            fname = "{}-{}".format(fname, str(args.gen_strategy))
        if args.checkpoint is not None:
            fname = "{}-checkpoint-{}".format(fname, step)
        if uid is not None:
            fname = "{}-id-{}".format(fname, uid)
        fname += ".json"
        
        logger.info("Saving results to {}".format(fname))
        with open(fname, "w") as fp:
            json.dump(results, fp, allow_nan=False)


def parse_args():
    parser = create_default_args()
    add_local_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))
    
    args = parse_args()
    logs_path = prepare_dirs(args)
    logger = logging.getLogger(__name__)
    main(args, logs_path)

