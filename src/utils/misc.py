import os
from typing import Optional
import contextlib
import joblib
from typing import Union
from itertools import chain

from yacs.config import CfgNode as CN
import mindspore as ms
from mindspore import Tensor
from mindspore import nn, ops


def lower_config(yacs_cfg):
    if not isinstance(yacs_cfg, CN):
        return yacs_cfg
    return {k.lower(): lower_config(v) for k, v in yacs_cfg.items()}


def upper_config(dict_cfg):
    if not isinstance(dict_cfg, dict):
        return dict_cfg
    return {k.upper(): upper_config(v) for k, v in dict_cfg.items()}


def flattenList(x):
    return list(chain(*x))


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument
    
    Usage:
        with tqdm_joblib(tqdm(desc="My calculation", total=10)) as progress_bar:
            Parallel(n_jobs=16)(delayed(sqrt)(i**2) for i in range(10))
            
    When iterating over a generator, directly use of tqdm is also a solutin (but monitor the task queuing, instead of finishing)
        ret_vals = Parallel(n_jobs=args.world_size)(
                    delayed(lambda x: _compute_cov_score(pid, *x))(param)
                        for param in tqdm(combinations(image_ids, 2),
                                          desc=f'Computing cov_score of [{pid}]',
                                          total=len(image_ids)*(len(image_ids)-1)/2))
    Src: https://stackoverflow.com/a/58936697
    """

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


class AllReduce(nn.Cell):
    def __init__(self, reduce: str = "mean", device_num: Optional[int] = None) -> None:
        super().__init__()
        self.average = reduce == "mean"

        if device_num is None:
            self.device_num = 1
        else:
            self.device_num = device_num

        self.all_reduce = ops.AllReduce()

    def construct(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        x = ops.cast(x, ms.float32)
        x = self.all_reduce(x)
        if self.average:
            x = x / self.device_num
        x = ops.cast(x, dtype)
        return x


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = Tensor(0.0, dtype=ms.float32)
        self.avg = Tensor(0.0, dtype=ms.float32)
        self.sum = Tensor(0.0, dtype=ms.float32)
        self.count = Tensor(0.0, dtype=ms.float32)

    def update(self, val: Tensor, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def fetch_optimizer_lr(opt):
    # print(f"Before, global step: {opt.global_step}")
    lr = opt.learning_rate
    if opt.dynamic_lr:
        if opt.is_group_lr:
            lr = ()
            for learning_rate in opt.learning_rate:
                # TODO: For ms2.1: opt.global_step -1 ms2.2: opt.global_step
                cur_dynamic_lr = learning_rate(opt.global_step).reshape(())
                lr += (cur_dynamic_lr,)
        else:
            lr = opt.learning_rate(opt.global_step - 1).reshape(())
    # print(f"After, global step: {opt.global_step}")
    return lr
