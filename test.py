import os
import sys
import time
import os.path as osp
import logging
import argparse
import pprint

import mindspore as ms
from tqdm import tqdm

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(__dir__, "..")))

from src.training.data import MultiSceneDataModule
from src.utils.logger import set_logger
from src.utils.seed import set_seed
from src.training.data_builder import build_dataset
from src.models import build_model
from src.config.default import get_cfg_defaults
from src.utils.metrics import compute_metrics, aggregate_metrics
from src.utils.misc import flattenList

logger = logging.getLogger("loftr.test")


def parse_args():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        'data_cfg_path', type=str, help='data config path')
    parser.add_argument(
        'main_cfg_path', type=str, help='main config path')
    parser.add_argument(
        '--ckpt_path', type=str, default=None,
        help='pretrained checkpoint path, helpful for using a pre-trained coarse-only LoFTR')
    parser.add_argument(
        '--dump_dir', type=str, default=None,
        help='dump directory')
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='training batch_size')
    parser.add_argument(
        '--num_workers', type=int, default=1,
        help='number of workers to read dataset')

    return parser.parse_args()


def main():
    args = parse_args()
    config = get_cfg_defaults()
    config.merge_from_file(args.main_cfg_path)
    config.merge_from_file(args.data_cfg_path)
    ms.set_context(mode=config.system.mode, device_id=config.system.device_id, device_target='Ascend')
    device_num = 1
    rank_id = 0

    # create logger, only rank0 log will be output to the screen
    set_logger(
        name="loftr",
        output_dir=config.TRAINER.ckpt_save_dir,
        rank=0,
        log_level=eval(config.get("log_level", "logging.INFO")),
    )
    if "DEVICE_ID" in os.environ:
        logger.info(
            f"Standalone testing. Device id: {os.environ.get('DEVICE_ID')}, "
            f"specified by environment variable 'DEVICE_ID'."
        )
    else:
        device_id = config.system.get("device_id", 0)
        ms.set_context(device_id=device_id)
        logger.info(
            f"Standalone testing. Device id: {device_id}, "
            f"specified by cfg.system.device_id in configs/loftr/outdoor/buggy_pos_env/loftr_ds.py"
            f" file or is default value 0."
        )

    set_seed(config.TRAINER.SEED)

    # create dataset
    data_module = MultiSceneDataModule(args, config)
    data_module.setup(stage='test', distribute=config.system.distribute)
    test_dataset, loader_test = build_dataset(
        data_module.test_dataset,
        data_module.test_loader_params,
        num_shards=device_num,
        shard_id=rank_id,
        is_train=False,
        refine_batch_size=True
    )

    # build model
    amp_level = config.system.get("amp_level", "O0")
    network, _ = build_model(config, pretrained_ckpt=args.ckpt_path,
                             amp_level=amp_level)
    network.set_train(False)

    # Infer
    num_batches = loader_test.get_dataset_size()
    data_cols = test_dataset.get_output_columns()
    data_iterator = loader_test.create_tuple_iterator(output_numpy=False, do_copy=False)

    output_metrics = []
    for in_data in tqdm(data_iterator, total=num_batches):
        batch_data = dict(zip(data_cols, in_data))
        model_input = []
        for col in ['image0', 'image1', 'mask0', 'mask1', 'scale0', 'scale1']:
            model_input.append(batch_data[col])
        match_kpts_f0, match_kpts_f1, match_conf, match_masks = network(*model_input)
        match_masks = match_masks.squeeze(0)
        num_valid_match = match_masks.sum()
        batch_data['m_bids'] = ms.Tensor([0 for _ in range(num_valid_match)], dtype=ms.int32)
        batch_data['mkpts0_f'] = match_kpts_f0.squeeze(0)[:num_valid_match]
        batch_data['mkpts1_f'] = match_kpts_f1.squeeze(0)[:num_valid_match]
        batch_data['pair_names'] = [eval(str(batch_data['pair_names_0'])), eval(str(batch_data['pair_names_1']))]
        metrics_batch, _ = compute_metrics(batch_data, config)
        output_metrics.append(metrics_batch)

    # Evaluate
    metrics = {k: flattenList([flattenList([_me[k] for _me in output_metrics])]) for k in output_metrics[0]}
    val_metrics_4tb = aggregate_metrics(metrics, config.TRAINER.EPI_ERR_THR)
    logger.info('\n' + pprint.pformat(val_metrics_4tb))


if __name__ == '__main__':
    main()
