#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

TRAIN_IMG_SIZE=640
# to reproduced the results in our paper, please use:
# TRAIN_IMG_SIZE=840
data_cfg_path="configs/data/megadepth_trainval_${TRAIN_IMG_SIZE}.py"
main_cfg_path="configs/loftr/outdoor/loftr_ds_dense.py"

n_nodes=1
batch_size=1
n_gpus_per_node=4
exp_name="outdoor-ds-${TRAIN_IMG_SIZE}-bs=$(($n_gpus_per_node * $n_nodes * $batch_size))"

python -u ./train.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name=${exp_name} \
    --num_nodes=${n_nodes} \
    --check_val_every_n_epoch=1 \
    --benchmark=True \
    --max_epochs=30 \
    --batch_size=${batch_size}
