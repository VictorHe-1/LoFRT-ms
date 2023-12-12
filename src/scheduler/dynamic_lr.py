import math
from bisect import bisect_right
from math import cos, pi


def linear_lr(start_factor, end_factor, total_iters, *, lr):
    lrs = []
    start_lr = lr * start_factor
    end_lr = lr * end_factor
    for i in range(total_iters):
        multiplier = i / total_iters
        lrs.append(start_lr + multiplier * (end_lr - start_lr))
    return lrs


def multi_step_lr(milestones, gamma, *, lr, steps_per_epoch, start_iter, total_iters):
    milestones = sorted(milestones)
    lrs = []
    for i in range(start_iter, total_iters):
        epoch_idx = math.floor(i / steps_per_epoch)
        lrs.append(lr * gamma ** bisect_right(milestones, epoch_idx))
    return lrs


def annealing_cos(start, end, factor, weight=1):
    """Calculate annealing cos learning rate.

    Cosine anneal from `weight * start + (1 - weight) * end` to `end` as
    percentage goes from 0.0 to 1.0.

    Args:
        start (float): The starting learning rate of the cosine annealing.
        end (float): The ending learing rate of the cosine annealing.
        factor (float): The coefficient of `pi` when calculating the current
            percentage. Range from 0.0 to 1.0.
        weight (float, optional): The combination factor of `start` and `end`
            when calculating the actual starting learning rate. Default to 1.
    """
    cos_out = cos(pi * factor) + 1
    return end + 0.5 * weight * (start - end) * cos_out


def get_warmup_lr(base_lr, cur_iters, warmup, warmup_ratio, warmup_iters):
    if warmup == 'constant':
        warmup_lr = base_lr * warmup_ratio
    elif warmup == 'linear':
        k = (1 - cur_iters / warmup_iters) * (1 -
                                              warmup_ratio)
        warmup_lr = base_lr * (1 - k)
    elif warmup == 'exp':
        k = warmup_ratio ** (1 - cur_iters / warmup_iters)
        warmup_lr = base_lr * k
    return warmup_lr


def cosine_annealing_lr(lr,
                        warmup,
                        warmup_iters,
                        warmup_ratio,
                        min_lr_ratio,
                        steps_per_epoch,
                        epochs):
    steps = steps_per_epoch * epochs
    lrs = []
    base_lr = lr
    target_lr = lr * min_lr_ratio
    regular_lr = None
    for i in range(steps):
        curr_iter = i
        curr_epoch = curr_iter // steps_per_epoch
        if curr_iter % steps_per_epoch == 0:  # before train epoch
            regular_lr = annealing_cos(base_lr, target_lr, curr_epoch / epochs)
        if warmup_iters > 0 and curr_iter < warmup_iters:  # before run iter
            lr = get_warmup_lr(regular_lr, curr_iter, warmup, warmup_ratio, warmup_iters)
            lrs.append(lr)
        else:
            lrs.append(regular_lr)
    return lrs


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    lrs = cosine_annealing_lr(0.0002,
                              'linear',
                              500,
                              1 / 3,
                              0.001,
                              242,
                              24)
    x = np.arange(len(lrs))
    plt.plot(x, lrs)
    plt.title('lr rate scheduler')
    plt.show()
