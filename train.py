"""
Model training
"""
import logging
import os
import shutil
import sys
import math
import argparse

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

import mindspore as ms
from mindspore.communication import get_group_size, get_rank, init

from src.config.default import get_cfg_defaults
from src.utils.logger import set_logger
from src.utils.seed import set_seed
from src.training.data import MultiSceneDataModule
from src.training.data_builder import build_dataset
from src.models import build_model
from src.utils.loss_scaler import get_loss_scales
from src.optimizers import build_optimizer, create_group_params
from src.scheduler import build_scheduler
from src.training.train_step_wrapper import TrainOneStepWrapper
from src.utils.ema import EMA
from src.utils.callbacks import EvalSaveCallback

logger = logging.getLogger("loftr.train")


def parse_args():
    # init a costum parser which will be added into pl.Trainer parser
    # check documentation: https://pytorch-lightning.readthedocs.io/en/latest/common/trainer.html#trainer-flags
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--exp_name', type=str, default='default_exp_name')
    parser.add_argument(
        '--num_nodes', type=int, default=1, help='number of PCs to use')
    parser.add_argument(
        '--check_val_every_n_epoch', type=int, default=1,
        help='check validation every n epoch')
    parser.add_argument(
        '--benchmark', type=bool, default=True,
        help='benchmark')
    parser.add_argument(
        '--max_epochs', type=int, default=30,
        help='number of epochs')
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='training batch_size')
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only LoFTR')
    parser.add_argument(
        '--parallel_load_data', action='store_true',
        help='load datasets in with multiple processes.')

    return parser.parse_args()


def main():
    # init env
    args = parse_args()
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    ms.set_context(mode=config.system.mode, device_target='Ascend')
    if config.system.distribute:
        init()
        device_num = get_group_size()
        rank_id = get_rank()
        ms.set_auto_parallel_context(
            device_num=device_num,
            parallel_mode="data_parallel",
            gradients_mean=True,
            # parameter_broadcast=True,
        )
        # create logger, only rank0 log will be output to the screen
        set_logger(
            name="loftr",
            output_dir=config.TRAINER.ckpt_save_dir,
            rank=rank_id,
            log_level=eval(config.get("log_level", "logging.INFO")),
        )
    else:
        device_num = 1
        rank_id = config.system.device_id

        # create logger, only rank0 log will be output to the screen
        set_logger(
            name="loftr",
            output_dir=config.TRAINER.ckpt_save_dir,
            rank=0,
            log_level=eval(config.get("log_level", "logging.INFO")),
        )
        if "DEVICE_ID" in os.environ:
            logger.info(
                f"Standalone training. Device id: {os.environ.get('DEVICE_ID')}, "
                f"specified by environment variable 'DEVICE_ID'."
            )
        else:
            device_id = config.system.get("device_id", 0)
            ms.set_context(mode=config.system.mode, device_id=device_id, device_target='Ascend')
            logger.info(
                f"Standalone training. Device id: {device_id}, "
                f"specified by system.device_id in yaml config file or is default value 0."
            )

    set_seed(config.TRAINER.SEED)

    # scale lr and warmup-step automatically
    config.TRAINER.WORLD_SIZE = device_num * args.num_nodes
    config.TRAINER.TRUE_BATCH_SIZE = config.TRAINER.WORLD_SIZE * args.batch_size
    _scaling = config.TRAINER.TRUE_BATCH_SIZE / config.TRAINER.CANONICAL_BS
    config.TRAINER.SCALING = _scaling
    config.TRAINER.TRUE_LR = config.TRAINER.CANONICAL_LR * _scaling
    config.TRAINER.WARMUP_STEP = math.floor(config.TRAINER.WARMUP_STEP / _scaling)

    # create dataset
    data_module = MultiSceneDataModule(args, config)
    data_module.setup(stage='fit', distribute=config.system.distribute)
    train_dataset, loader_train = build_dataset(
        data_module.train_dataset,
        data_module.train_loader_params,
        num_shards=device_num,
        shard_id=rank_id,
        is_train=True,
    )
    val_dataset, loader_eval = build_dataset(
        data_module.val_dataset,
        data_module.val_loader_params,
        num_shards=device_num,
        shard_id=rank_id,
        is_train=False,
        refine_batch_size=True
    )
    num_batches = loader_train.get_dataset_size()

    # create model
    amp_level = config.system.get("amp_level", "O0")
    network = build_model(config, pretrained_ckpt=args.ckpt_path,
                          amp_level=amp_level, training_mode=True)
    num_params = sum([param.size for param in network.get_parameters()])
    num_trainable_params = sum([param.size for param in network.trainable_params()])
    # get loss scale setting for mixed precision training
    loss_scale_manager, optimizer_loss_scale = get_loss_scales(config)
    # build lr scheduler
    lr_scheduler = build_scheduler(num_batches,
                                   config.TRAINER.SCHEDULER,
                                   lr=config.TRAINER.TRUE_LR,
                                   warmup_iters=config.TRAINER.WARMUP_STEP,
                                   warmup_factor=config.TRAINER.WARMUP_RATIO,
                                   decay_rate=config.TRAINER.MSLR_GAMMA,
                                   milestones=config.TRAINER.MSLR_MILESTONES,
                                   num_epochs=args.max_epochs)
    # build optimizer
    # cfg.optimizer.update({"lr": lr_scheduler, "loss_scale": optimizer_loss_scale})
    params = create_group_params(network.trainable_params())  # TODO: currently no param grouping, confirm param grouping

    # this setting doesn't take effect, just keep it.
    weight_decay = config.TRAINER.ADAMW_DECAY if config.TRAINER.ADAMW_DECAY else config.TRAINER.ADAM_DECAY
    optimizer = build_optimizer(params, config, lr_scheduler, weight_decay=weight_decay, filter_bias_and_bn=False)  # TODO: confirm filter_bias_and_bn
    # resume ckpt
    start_epoch = 0
    # build train step cell
    gradient_accumulation_steps = config.TRAINER.get("gradient_accumulation_steps", 1)
    clip_grad = config.TRAINER.get("clip_grad", False)
    use_ema = config.TRAINER.get("ema", False)
    ema = EMA(network, ema_decay=config.TRAINER.get("ema_decay", 0.9999), updates=0) if use_ema else None

    # input_idx meaning:
    # img0, img1, mask_c0, mask_c1, scale_0, scale_1,
    # conf_matrix_gt, spv_w_pt0_i, spv_pt1_i, spv_b_ids, spv_i_ids, spv_j_ids,
    train_net = TrainOneStepWrapper(
        network,
        optimizer=optimizer,
        scale_sense=loss_scale_manager,
        drop_overflow_update=config.system.drop_overflow_update,
        gradient_accumulation_steps=gradient_accumulation_steps,
        clip_grad=clip_grad,
        clip_norm=config.TRAINER.get("clip_norm", 1.0),
        ema=ema,
        config=config,
        data_cols=train_dataset.get_output_columns(),
        input_idx=[0, 2, 15, 16, 8, 9, 17, 18, 19, 20, 21, 22]
    )

    # build callbacks
    eval_cb = EvalSaveCallback(
        network,
        loader_eval,
        config,
        rank_id=rank_id,
        device_num=device_num,
        batch_size=args.batch_size,
        ckpt_save_dir=config.TRAINER.ckpt_save_dir,
        main_indicator=config.metrics.main_indicator,
        ema=ema,
        loader_output_columns=val_dataset.get_output_columns(),
        input_indices=[0, 2, 15, 16, 8, 9],  # img0, img1, mask_c0, mask_c1, scale_0, scale_1
        val_interval=config.system.get("val_interval", 1),
        val_start_epoch=config.system.get("val_start_epoch", 1),
        log_interval=config.system.get("log_interval", 1),
        ckpt_save_policy=config.system.get("ckpt_save_policy", "top_k"),
        ckpt_max_keep=config.system.get("ckpt_max_keep", 10),
        start_epoch=start_epoch
    )

    # log
    num_devices = device_num if device_num is not None else 1
    global_batch_size = args.batch_size * num_devices * gradient_accumulation_steps
    model_name = "loFTR"
    info_seg = "=" * 40
    weight_decay = config.TRAINER.get('ADAMW_DECAY',
                                      config.TRAINER.get('ADAM_DECAY', 0))
    logger.info(
        f"\n{info_seg}\n"
        f"Distribute: {config.system.distribute}\n"
        f"Model: {model_name}\n"
        f"Total number of parameters: {num_params}\n"
        f"Total number of trainable parameters: {num_trainable_params}\n"
        f"Optimizer: {config.TRAINER.OPTIMIZER}\n"
        f"Weight decay: {weight_decay} \n"
        f"Batch size: {args.batch_size}\n"
        f"Num devices: {num_devices}\n"
        f"Gradient accumulation steps: {gradient_accumulation_steps}\n"
        f"Global batch size: {args.batch_size}x{num_devices}x{gradient_accumulation_steps}="
        f"{global_batch_size}\n"
        f"LR: {config.TRAINER.TRUE_LR} \n"
        f"Scheduler: {config.TRAINER.SCHEDULER}\n"
        f"Steps per epoch: {num_batches}\n"
        f"Num epochs: {args.max_epochs}\n"
        f"Clip gradient: {clip_grad}\n"
        f"EMA: {use_ema}\n"
        f"AMP level: {amp_level}\n"
        f"Loss scaler: {config.get('loss_scaler', None)}\n"  # TODO
        f"Drop overflow update: {config.system.drop_overflow_update}\n"
        f"{info_seg}\n"
        f"\nStart training... (The first epoch takes longer, please wait...)\n"
    )
    # training
    model = ms.Model(train_net)
    model.train(
        args.max_epochs,
        loader_train,
        callbacks=[eval_cb],
        dataset_sink_mode=config.TRAINER.dataset_sink_mode,
        initial_epoch=start_epoch,
    )


if __name__ == "__main__":
    main()
