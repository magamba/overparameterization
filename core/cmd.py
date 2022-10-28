# -*- coding: utf-8 -*-

"""
A series of common functions for dealing with command line arguments.

The precedence for arguments is
    environment variables --> command line
with command line having the highest priority. If options are in the environment,
they can be overridden.
"""

from argparse import ArgumentParser, Action
from os import environ
from functools import wraps

import core.data as dsets
from models.concepts import ALL_ADDITIONS
from models.factory import MODEL_FACTORY_MAP
from core.random_dataset import __random_datasets_all__ as random_dataset_choices

Datasets = dsets.Datasets

def ensure(ensure_func, n=1):
    def decorator_ensure(func):
        @wraps(func)
        def wrapper_ensure(self, *args, **kwargs):
            ensure_func(self, args[:n])
            return func(self, *args, **kwargs)

        return wrapper_ensure

    return decorator_ensure


class ReusableArgumentParser(ArgumentParser):
    def __init__(self, *args, **kwargs):
        self.opts_to_acts = {}
        self.injections = {}
        self.conditional_injections = {}
        self.set_op_to_require_op = {}
        super().__init__(*args, **kwargs)

    def add_argument(self, *args, **kwargs):
        opt_action = super().add_argument(*args, **kwargs)
        self.opts_to_acts[opt_action.dest] = opt_action
        return opt_action

    def parse_args(self, *args, **kwargs):
        parsed_args = super().parse_args(*args, **kwargs)
        self._set_opts_after_parsed(parsed_args, self.injections)

        for cond_injection_key in self.conditional_injections:
            cond_val, injections = self.conditional_injections[cond_injection_key]
            if getattr(parsed_args, cond_injection_key) == cond_val:
                self._set_opts_after_parsed(parsed_args, injections)
        for set_op, (is_set_func, require_op) in self.set_op_to_require_op.items():
            set_val = getattr(parsed_args, set_op)
            require_val = getattr(parsed_args, require_op)
            if is_set_func(set_val, require_val):
                raise ValueError(
                    "Under {}, {}={} but {}={}".format(
                        is_set_func, set_op, set_val, require_op, require_val
                    )
                )
        return parsed_args

    @staticmethod
    def _set_opts_after_parsed(parsed_args, injections):
        for injection_key in injections:
            setattr(parsed_args, injection_key, injections[injection_key])

    def _require_opt_in_parser(self, opt_names):
        for opt_name in opt_names:
            if opt_name not in self.opts_to_acts:
                raise ValueError("{} is not in parser".format(opt_name))

    @ensure(_require_opt_in_parser)
    def set_required(self, opt_name, required):
        self.opts_to_acts[opt_name].required = required

    @ensure(_require_opt_in_parser)
    def inject_opt(self, opt_name, opt_value):
        self.set_required(opt_name, False)
        self.injections[opt_name] = opt_value

    @ensure(_require_opt_in_parser)
    def if_set(self, opt_name, opt_val, injections):
        self.conditional_injections[opt_name] = (opt_val, injections)

    @ensure(_require_opt_in_parser, 2)
    def if_set_require(self, set_opt, required_opt):
        self.if_set_require_under(set_opt, required_opt, lambda s, r: s and not r)

    @ensure(_require_opt_in_parser, 2)
    def if_set_require_under(self, set_opt, required_opt, is_set_func):
        if not callable(is_set_func):
            raise ValueError("{} is not a function".format(is_set_func))
        self.set_op_to_require_op[set_opt] = (is_set_func, required_opt)


class DefaultToEnvOpt(Action):
    def __init__(
        self,
        option_strings,
        dest,
        const=None,
        required=False,
        default=None,
        type=None,
        **kwargs
    ):
        env_name = const if const else dest.upper()
        # First check ENV, then can overwrite with command line, fallback is the default
        old_default = default
        default = environ.get(env_name) or old_default
        if type:
            default = type(default)
        # if the opt is required, but we found it in environ, then the user doesn't
        # have to specify it
        if required and default is not None:
            required = False

        if default is None:
            default = old_default

        super().__init__(
            option_strings,
            dest,
            const=const,
            default=default,
            required=required,
            type=type,
            **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, values)


class DefaultToEnvFlag(DefaultToEnvOpt):
    def __init__(
        self, option_strings, dest, nargs=None, type=None, required=None, **kwargs
    ):
        super().__init__(
            option_strings, dest, nargs=0, type=bool, required=False, **kwargs
        )

    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, True)


def add_env_opts(parser):
    parser.add_argument(
        "--device",
        action=DefaultToEnvOpt,
        default="cpu",
        help="Should follow PyTorch standards (i.e. cpu, cuda, cuda:1, etc.)",
    )
    parser.add_argument(
        "--name",
        help="the name of this experimental run, a subdirectory under --SAVE-DIR",
        action=DefaultToEnvOpt,
    )
    parser.add_argument(
        "--save-dir",
        default="./",
        help="the top level directory to save all logs to",
        action=DefaultToEnvOpt,
    )
    parser.add_argument(
        "--data-dir",
        default="",
        action=DefaultToEnvOpt,
        help="the top level directory where datasets are loaded from",
    )
    parser.add_argument(
        "--workers",
        type=int,
        action=DefaultToEnvOpt,
        default=0,
        help="Number of dataloader worker processes to spawn",
    )


def add_default_opts(parser):
    parser.add_argument(
        "--data",
        choices=[str(d) for d in dsets.DATASETS],
        help="which data set to use",
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_FACTORY_MAP.keys()),
        type=str,
        default=None,
        help="the model to use, see models/factory.py#MODEL_FACTORY_MAP for options",
    )
    parser.add_argument(
        "--model-additions",
        choices=ALL_ADDITIONS,
        nargs="*",
        default=(),
        help="network architecture additions, e.g. batch_norm and dropout",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--augmentation",
        default=False,
        action="store_true",
        help="enable (weak) data augmentation",
    )
    parser.add_argument(
        "--label-noise",
        default=0.,
        type=float,
        help="the fraction of corrupted training labels",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="seed used to control weight initialization and shuffling of batches"
    )
    parser.add_argument(
        "--label-seed", type=int, default=None, help="seed used for corrupting labels"
    )
    parser.add_argument(
        "--data-split-seed", type=int, default="42", help="seed used for making a validation split out of the train set"
    )
    parser.add_argument(
        "--train-split",
        type=int,
        default=None,
        help="the number of samples in the train set. the sum of this"
        "and --val-split should equal the length of the true"
        "train set. If setting, --val-split must also be set.",
    )
    parser.add_argument(
        "--val-split",
        type=int,
        default=None,
        help="the number of samples in the validation set. the sum of"
        "this and --train-split should equal the length of the "
        "true train set. Must provide in conjunction with "
        "--train-split",
    )
    parser.if_set_require("train_split", "val_split")
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        help="log level",
    )
    parser.add_argument(
        "--logfile",
        type=str,
        default="output.log",
        help="file name to save logs to",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.,
        help="value for L2 weight decay in conjunction with SGD",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.,
        help="dropout rate",
    )
    parser.add_argument(
        "--l1-regularization",
        type=float,
        default=None,
        help="L1-regularization coefficient, may use in conjunction "
        "with --l2-regularization for elastic net regularization",
    )
    parser.add_argument(
        "--l2-regularization",
        type=float,
        default=None,
        help="L2-regularization coefficient, may use in conjunction "
        "with --l1-regularization for elastic net regularization",
    )
    parser.add_argument(
        "--random-dataset",
        type=str,
        default=None,
        choices=random_dataset_choices,
        help="Use in combination with DATA. Generate random dataset with the same pixel-wise statistics as DATA.",
    )


def create_default_args():
    parser = ReusableArgumentParser()
    add_env_opts(parser)
    add_default_opts(parser)
    return parser


def create_env_args():
    parser = ReusableArgumentParser()
    add_env_opts(parser)
    return parser

