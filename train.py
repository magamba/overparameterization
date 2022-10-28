# -*- coding: utf-8 -*-

"""
Training script
"""

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import MultiplicativeLR
from ignite.engine import Events, create_supervised_trainer, create_supervised_evaluator
from ignite.metrics import Accuracy, Loss
from ignite.utils import setup_logger
from ignite.handlers import Checkpoint, DiskSaver
from ignite.handlers.param_scheduler import LinearCyclicalScheduler, LRScheduler, create_lr_scheduler_with_warmup
from ignite.contrib.handlers.tensorboard_logger import (
    TensorboardLogger,
    OutputHandler,
    OptimizerParamsHandler,
)

from shutil import make_archive, rmtree
import sys
import os
from os import path, environ
from typing import Union, List, Tuple, Callable
import zipfile
import pickle
import logging

from core.cmd import create_default_args
from core.utils import (
    init_torch,
    global_iteration_from_engine,
    all_log_dir,
    checkpoint_dir,
    init_logging,
    init_prngs,
    prepare_dirs
)
from core.data import create_data_manager
from models import factory as model_factory
from interface.handlers import StopOnInterpolateByAccuracy, StopOnInterpolateByLoss

logger = logging.getLogger(__name__)


def setup_tb_logger(
    args, trainer, evaluator, test_evaluator, val_evaluator, optimizer
):
    with SummaryWriter(log_dir=all_log_dir(args)) as writer:
        writer.add_text("data", args.data)
        writer.add_text("model", args.model)
        writer.add_text("model_additions", ",".join(args.model_additions))
        writer.add_text("learning_rate", str(args.learning_rate))
        writer.add_text("batch_size", str(args.batch_size))
        writer.add_text("epochs", str(args.epochs))
        writer.add_text("seed", str(args.seed))
        writer.add_text("optimizer", args.optimizer)
        writer.add_text("early_exit_accuracy", str(args.early_exit_accuracy))
        writer.add_text("dropout", str(args.dropout))
        writer.add_text("early_exit_loss", str(args.early_exit_loss))
        writer.add_text("lr_decay_rate", str(args.lr_decay_rate))
        writer.add_text("lr_step", str(args.lr_step))
        writer.add_text("stop_by_loss_threshold", str(args.stop_by_loss_threshold))
        writer.add_text(
            "stop_by_accuracy_threshold", str(args.stop_by_accuracy_threshold)
        )

        writer.add_text("DEVICE", args.device)
        writer.add_text("NAME", args.name)
        writer.add_text("SAVE_DIR", args.save_dir)
        writer.add_text("DATA_DIR", args.data_dir)
        writer.add_text("WORKERS", str(args.workers))

    logger = TensorboardLogger(log_dir=all_log_dir(args))
    logger.attach(
        trainer,
        log_handler=OutputHandler(
            tag="train",
            output_transform=lambda loss: {"batch_loss": loss},
            metric_names="all",
        ),
        event_name=Events.ITERATION_COMPLETED(every=args.loss_checkpoint),
    )

    logger.attach(
        evaluator,
        log_handler=OutputHandler(
            tag="train",
            metric_names=["loss", "accuracy"],
            global_step_transform=global_iteration_from_engine(trainer),
        ),
        event_name=Events.COMPLETED,
    )
    logger.attach(
        test_evaluator,
        log_handler=OutputHandler(
            tag="test",
            metric_names=["loss", "accuracy"],
            global_step_transform=global_iteration_from_engine(trainer),
        ),
        event_name=Events.COMPLETED,
    )
    if val_evaluator is not None:
        logger.attach(
            val_evaluator,
            log_handler=OutputHandler(
                tag="validation",
                metric_names=["loss", "accuracy"],
                global_step_transform=global_iteration_from_engine(trainer),
            ),
            event_name=Events.COMPLETED,
        )
    logger.attach(
        trainer,
        log_handler=OptimizerParamsHandler(optimizer),
        event_name=Events.ITERATION_STARTED,
    )

    return logger


def every_and_n_times(every, n):
    if every is None:
        every = 1
    if n is None:
        n = float("inf")

    def wrap_every_and_n_times(engine_, event_num):
        times_called = wrap_every_and_n_times.times_called
        wrap_every_and_n_times.times_called += 1
        if every == 0:
            return False
        if event_num % every == 0 and times_called < n:
            return True
        return False

    wrap_every_and_n_times.times_called = 0
    return wrap_every_and_n_times


def _every_or_specified(save_checkpoints: Union[List, Tuple]) -> Callable:
    def _never_event_filter(engine_, step):
        return False

    def _every_event_filter(engine_, step):
        return step % save_checkpoints[0] == 0

    def _specified_event_filter(engine_, step):
        if step in save_checkpoints:
            return True
        return False

    if type(save_checkpoints) == int:
        save_checkpoints = (save_checkpoints,)

    if len(save_checkpoints) == 0:
        return _never_event_filter
    elif len(save_checkpoints) == 1:
        return _never_event_filter if save_checkpoints[0] == 0 else _every_event_filter
    return _specified_event_filter


def run(args, data_manager, model, device):
    data_loader = data_manager.dloader
    num_iterations = len(data_loader)
    if args.optimizer == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
            momentum=args.momentum,
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay
        )
    loss_fn = torch.nn.CrossEntropyLoss().to(device)
    if args.l1_regularization:
        loss_fn = loss_fn_with_regularization(
            loss_fn, model, args.l1_regularization, 1
        )
    if args.l2_regularization:
        loss_fn = loss_fn_with_regularization(
            loss_fn, model, args.l2_regularization, 2
        )
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device)
    evaluator = create_supervised_evaluator(
        model, metrics={"accuracy": Accuracy(), "loss": Loss(loss_fn)}, device=device
    )
    test_evaluator = create_supervised_evaluator(
        model, metrics={"accuracy": Accuracy(), "loss": Loss(loss_fn)}, device=device
    )
    val_evaluator = None
    if data_manager.vloader is not None:
        val_evaluator = create_supervised_evaluator(
            model,
            metrics={"accuracy": Accuracy(), "loss": Loss(loss_fn)},
            device=device,
        )
    trainer.logger = setup_logger("trainer", level=logging.INFO)

    def compute_metrics(engine_):
        evaluator.run(data_loader)
        test_evaluator.run(data_manager.tloader)
        if data_manager.vloader is not None:
            val_evaluator.run(data_manager.vloader)

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(every=args.eval_every * num_iterations), compute_metrics
    )
    trainer.add_event_handler(Events.COMPLETED, compute_metrics)
    
    if args.early_exit_accuracy:
        evaluator.add_event_handler(
            Events.COMPLETED,
            StopOnInterpolateByAccuracy(threshold=args.stop_by_accuracy_threshold),
            trainer,
        )
    elif args.early_exit_loss:
        evaluator.add_event_handler(
            Events.COMPLETED,
            StopOnInterpolateByLoss(threshold=args.stop_by_loss_threshold),
            trainer,
        )
    lr_scheduler = None

    def _lr_mult(epoch):
        if args.lr_step == 0:
            return 1  # constant lr
        if args.lr_decay_rate == 0:
            return 1
        if (epoch % args.lr_step == 0) or (
            epoch % int(args.epochs * 0.75) == 0
        ):
            return args.lr_decay_rate
        return 1

    if args.lr_step > 0:
        logger.info("Setting up learning rate scheduler. Lr: {}, step: {}, decay mult {}.".format(args.learning_rate, args.lr_step, args.lr_decay_rate))
        lr_scheduler = LRScheduler(MultiplicativeLR(optimizer, lr_lambda=_lr_mult))
        trainer.add_event_handler(Events.EPOCH_COMPLETED, lr_scheduler)
    elif args.lr_warmup_epochs > 0:
        logger.info("Setting up learning rate scheduler with warmup. Lr: {}, Start lr: {}, Warmup steps: {}.".format(args.learning_rate, args.lr_warmup_initial, args.lr_warmup_epochs))
        lr_scheduler = create_lr_scheduler_with_warmup(
            LRScheduler(MultiplicativeLR(optimizer, lr_lambda=_lr_mult)),
            warmup_start_value=args.lr_warmup_initial,
            warmup_duration=args.lr_warmup_epochs,
            warmup_end_value=args.learning_rate
        )
        trainer.add_event_handler(Events.EPOCH_STARTED, lr_scheduler)
    elif args.optimizer == "sgd":
        logger.info("Training with constant learning rate: {}".format(args.learning_rate))

    tb_logger = setup_tb_logger(
        args, trainer, evaluator, test_evaluator, val_evaluator, optimizer
    )

    objects_to_checkpoint = {"trainer": trainer, "model": model, "optimizer": optimizer}
    if lr_scheduler:
        objects_to_checkpoint["lr_scheduler"] = lr_scheduler
    
    training_checkpoint = Checkpoint(
        to_save=objects_to_checkpoint,
        save_handler=DiskSaver(checkpoint_dir(args), require_empty=False),
        n_saved=None,
    )

    trainer.add_event_handler(
        Events.ITERATION_COMPLETED(
            event_filter=_every_or_specified(
                tuple(it * num_iterations for it in args.save_checkpoint)
            )
        ),
        training_checkpoint,
    )

    if args.resume_from is not None:
        chkpt_dir = checkpoint_dir(args)
        chkpt_zip = "{}.zip".format(chkpt_dir)
        logger.info("Resuming from checkpoint {} in {}".format(args.resume_from, chkpt_zip))
        if os.path.isfile(chkpt_zip):
            load_from_zipped_checkpoint(
                objects_to_checkpoint, chkpt_zip, args.resume_from
            )
        else:
            load_from_checkpoint(
                objects_to_checkpoint, chkpt_dir, args.resume_from
            )
    logger.info("Running")
    trainer.add_event_handler(Events.EPOCH_STARTED(once=1), training_checkpoint)
    logger.info("Starting training")
    trainer.run(data_loader, max_epochs=args.epochs)
    # save a checkpoint at the end. if the accuracy is 100%, @run will be stopped
    # and the current model may or may not be saved
    logger.info("Stopped training")
    training_checkpoint(trainer)

    tb_logger.close()


def load_from_checkpoint(objects_to_checkpoint, chkpt_dir, resume_from):
    checkpoint = torch.load("{}/checkpoint_{}.pt".format(chkpt_dir, resume_from))
    Checkpoint.load_objects(to_load=objects_to_checkpoint, checkpoint=checkpoint)


def load_from_zipped_checkpoint(objects_to_checkpoint, chkpt_zip, resume_from):
    chkpt_dir = os.path.join(os.path.dirname(chkpt_zip), "checkpoints")
    logger.info("Checkpoint dir: {}".format(chkpt_dir))
    with zipfile.ZipFile(chkpt_zip, "r") as zdir:
        logger.info("Extracting {} to {}".format(chkpt_zip, chkpt_dir))
        zdir.extractall(path=chkpt_dir)
    checkpoint = torch.load(
        os.path.join(chkpt_dir, "checkpoint_{}.pt".format(resume_from))
    )
    Checkpoint.load_objects(
        to_load=objects_to_checkpoint, checkpoint=checkpoint
    )


def compress_checkpoints(args):
    compress_dir = checkpoint_dir(args)
    make_archive(compress_dir, "zip", compress_dir)
    rmtree(compress_dir)


def add_local_args(parser):
    opt_group = parser.add_argument_group("train local")
    opt_group.add_argument(
        "--epochs",
        type=int,
        help="max number of epochs to train for",
        default=300,
    )
    opt_group.add_argument(
        "--learning-rate",
        type=float,
        default=0.1,
        help="learning rate for training",
    )
    opt_group.add_argument(
        "--lr-step",
        "--learning-rate-step",
        type=int,
        default=0,
        help="Every --lr-step, modify the learning rate. Set to 0 to disable.",
    )
    opt_group.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam"],
        help="optimizer to use for training",
    )
    opt_group.add_argument(
        "--momentum", type=float, default=0.9, help="momentum coefficient for SGD"
    )
    opt_group.add_argument(
        "--eval-every",
        type=int,
        default=1,
        help="compute metrics (loss, accuracy) every EVAL_EVERY epochs",
    )
    opt_group.add_argument(
        "--early-exit-accuracy",
        action="store_true",
        help="true to stop training once accuracy is 1.0",
    )
    opt_group.add_argument(
        "--early-exit-loss",
        action="store_true",
        help="stop training based on training loss threshold",
    )
    opt_group.add_argument(
        "--lr-decay-rate",
        type=float,
        default=0.,
        help="the value to use for multiplicative learning reate "
        "decay. set to 0. for no lr decay",
    )
    
    opt_group.add_argument(
        "--lr-warmup-epochs",
        type=int,
        default=0,
        help="Number of training epochs for learning rate warmup",
    )
    
    opt_group.add_argument(
        "--lr-warmup-initial",
        type=float,
        default=0.001,
        help="The starting value for the learning rate warmup.",
    )
    
    opt_group.add_argument(
        "--stop-by-loss-threshold",
        type=float,
        default=0.19,
        help="the training loss to stop training at. must set "
        "--early-exit-loss as well to true in conjunction",
    )
    opt_group.add_argument(
        "--stop-by-accuracy-threshold",
        type=float,
        default=0.0,
        help="the training accuracy to stop training at. must set "
        "--early-exit-accuracy to true in conjunction",
    )   
    opt_group.add_argument(
        "--resume-from",
        type=int,
        default=None,
        help="Checkpoint to resume training from, if interrupted previously.",
    )
    opt_group.add_argument(
        "--save-checkpoint",
        nargs="*",
        default=(0,),
        type=int,
        help="When to save model checkpoints to disk, expressed in epochs. By default a checkpoint of the model at initialization and at convergence are saved. If one value is specified, it denotes the checkpoint frequency. If multiple values are given, they are used as explicit checkpoints.",
    )
    # @deprecate
    opt_group.add_argument("--accuracy-threshold", type=float, default=1.0)
    opt_group.add_argument(
        "--loss-checkpoint",
        type=int,
        default=382,
        help="Report/save the batch loss every L_LOSS-CHECKPOINT iterations.",
    )
    opt_group.add_argument(
        "--raw-checkpoints",
        help="If true, do not zip the model checkpoints into one zip file.",
    )
    opt_group.add_argument(
        "--weight-init",
        type=str,
        default="pytorch",
        choices=["fixup", "mfixup", "nkaiming", "pytorch"],
        help="Weight initialization scheme to use (default = pytorch).",
    )


def main(args):
    logger = logging.getLogger(__name__)

    logger.info(args)
    init_torch(cmd_args=args, double_precision=False)
    init_prngs(args)

    use_bn = True if 'batch_norm' in args.model_additions else False
    model = model_factory.create_model(
        args.model, args.data, additions=args.model_additions, dropout_rate=args.dropout, use_batch_norm=use_bn, weight_init=args.weight_init
    )
    
    if torch.cuda.is_available() and args.device == "cuda":
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    model.train()
    model.to(device)
    tv_split = (None, None)
    if args.train_split and args.val_split:
        tv_split = (args.train_split, args.val_split)
    logger.info("Train val split {}".format(tv_split))
    run(
        args,
        create_data_manager(
            args,
            args.label_noise,
            augment=args.augmentation,
            seed=args.label_seed,
            train_validation_split=tv_split,
        ),
        model,
        device
    )
    if not args.raw_checkpoints:
        compress_checkpoints(args)


def parse_args():
    parser = create_default_args()
    add_local_args(parser)
    return parser.parse_args()


if __name__ == "__main__":
    import signal
    signal.signal(signal.SIGINT, lambda x, y: sys.exit(0))

    args = parse_args()
    prepare_dirs(args)
    main(args)
