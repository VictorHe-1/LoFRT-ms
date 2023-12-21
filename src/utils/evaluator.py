import logging
from typing import List

from tqdm import tqdm

import mindspore as ms
from mindspore.common import dtype as mstype
from mindspore.ops import functional as F

from .metrics import compute_metrics, aggregate_metrics
from .misc import flattenList


__all__ = ["Evaluator"]
_logger = logging.getLogger(__name__)


class Evaluator:
    """
    Args:
        network: network
        dataloader : data loader to generate batch data, where the data columns in a batch are defined by the transform
            pipeline and `output_columns`.
        input_indices: The indices of the data tuples which will be fed into the network.
            If it is None, then the first item will be fed only.
    """

    def __init__(
        self,
        network,
        dataloader,
        loader_output_columns=None,
        input_indices=None,
        config=None
    ):
        self.net = network
        # create iterator
        self.reload(
            dataloader,
            loader_output_columns,
            input_indices
        )
        self.config = config
        self.metric_names = ['prec@5e-4', 'auc@5', 'auc@10', 'auc@20']

    def reload(
        self,
        dataloader,
        loader_output_columns=None,
        input_indices=None
    ):
        # create iterator
        self.iterator = dataloader.create_tuple_iterator(output_numpy=False, do_copy=False)
        self.num_batches_eval = dataloader.get_dataset_size()

        # dataset output columns
        self.loader_output_columns = loader_output_columns or []
        self.input_indices = input_indices

    def eval(self):
        """
        Args:
        """
        self.net.set_train(False)
        output_metrics = []
        for i, data in tqdm(enumerate(self.iterator), total=self.num_batches_eval):
            if self.input_indices is not None:
                inputs = [data[x] for x in self.input_indices]
            else:
                inputs = [data[0]]
            # For testing: we don't need spv_b_ids, spv_i_ids, spv_j_ids
            inputs.extend([None, None, None])
            match_kpts_f0, match_kpts_f1, match_conf, match_masks = self.net(*inputs)
            batch_data = dict(zip(self.loader_output_columns, in_data))
            match_masks = match_masks.squeeze(0)
            num_valid_match = match_masks.sum()
            batch_data['m_bids'] = ms.Tensor([0 for _ in range(num_valid_match)], dtype=ms.int32)
            batch_data['mkpts0_f'] = match_kpts_f0.squeeze(0)[:num_valid_match]
            batch_data['mkpts1_f'] = match_kpts_f1.squeeze(0)[:num_valid_match]
            batch_data['pair_names'] = [eval(str(batch_data['pair_names_0'])), eval(str(batch_data['pair_names_1']))]
            metrics_batch, _ = compute_metrics(batch_data, self.config)
            output_metrics.append(metrics_batch)

        metrics = {k: flattenList([flattenList([_me[k] for _me in output_metrics])]) for k in output_metrics[0]}
        val_metrics_4tb = aggregate_metrics(metrics, self.config.TRAINER.EPI_ERR_THR)

        self.net.set_train(True)

        return val_metrics_4tb
