"""Scheduler Factory"""
import logging

from .dynamic_lr import (
    multi_step_lr,
    linear_lr
)

__all__ = ["build_scheduler"]
_logger = logging.getLogger(__name__)


def build_scheduler(
    steps_per_epoch: int,
    scheduler: str = "constant",
    lr: float = 0.01,
    warmup_iters: int = 3,
    warmup_factor: float = 0.0,
    decay_rate: float = 0.9,
    milestones: list = None,
    num_epochs: int = 10
):
    r"""Creates learning rate scheduler by name.

    Args:
        steps_per_epoch: number of steps per epoch.
        scheduler: scheduler name like 'constant', 'cosine_decay', 'step_decay',
            'exponential_decay', 'polynomial_decay', 'multi_step_decay'. Default: 'constant'.
        lr: learning rate value. Default: 0.01.
        min_lr: lower lr bound for 'cosine_decay' schedulers. Default: 1e-6.
        warmup_epochs: epochs to warmup LR, if scheduler supports. Default: 3.
        warmup_factor: the warmup phase of scheduler is a linearly increasing lr,
            the beginning factor is `warmup_factor`, i.e., the lr of the first step/epoch is lr*warmup_factor,
            and the ending lr in the warmup phase is lr. Default: 0.0
        decay_epochs: for 'cosine_decay' schedulers, decay LR to min_lr in `decay_epochs`.
            For 'step_decay' scheduler, decay LR by a factor of `decay_rate` every `decay_epochs`. Default: 10.
        decay_rate: LR decay rate (default: 0.9)
        milestones: list of epoch milestones for 'multi_step_decay' scheduler. Must be increasing.
        num_epochs: number of total epochs.
        lr_epoch_stair: If True, LR will be updated in the beginning of each new epoch
            and the LR will be consistent for each batch in one epoch.
            Otherwise, learning rate will be updated dynamically in each step. (default=False)
    Returns:
        A list of float numbers indicating the learning rate at every step
    """
    # check params
    if milestones is None:
        milestones = []

    if warmup_iters > steps_per_epoch * num_epochs:
        _logger.warning("warmup_epochs + decay_epochs > num_epochs. Please check and reduce decay_epochs!")

    # lr warmup phase
    warmup_lr_scheduler = []
    if warmup_iters > 0:
        if warmup_factor == 0:
            _logger.warning(
                "The warmup factor is set to 0, lr of 0-th epoch is always zero! " "Recommend value is 0.01."
            )
        warmup_func = linear_lr
        warmup_lr_scheduler = warmup_func(
            start_factor=warmup_factor,
            end_factor=1.0,
            total_iters=warmup_iters,
            lr=lr
        )

    # lr decay phase
    total_iters = steps_per_epoch * num_epochs
    main_iters = total_iters - warmup_iters
    if scheduler == "MultiStepLR":
        main_lr_scheduler = multi_step_lr(
            milestones=milestones,
            gamma=decay_rate,
            lr=lr,
            steps_per_epoch=steps_per_epoch,
            start_iter=warmup_iters,
            total_iters=total_iters
        )
    elif scheduler == "constant":
        main_lr_scheduler = [lr for _ in range(main_iters)]
    else:
        raise ValueError(f"Invalid scheduler: {scheduler}")

    # combine
    lr_scheduler = warmup_lr_scheduler + main_lr_scheduler

    return lr_scheduler
