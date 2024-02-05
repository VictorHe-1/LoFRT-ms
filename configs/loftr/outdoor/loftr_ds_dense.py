from src.config.default import _CN as cfg

cfg.LOFTR.MATCH_COARSE.MATCH_TYPE = 'dual_softmax'
cfg.LOFTR.MATCH_COARSE.SPARSE_SPVS = False

cfg.TRAINER.CANONICAL_LR = 8e-3
cfg.TRAINER.WARMUP_STEP = 1875  # 3 epochs
cfg.TRAINER.WARMUP_RATIO = 0.1
cfg.TRAINER.MSLR_MILESTONES = [8, 12, 16, 20, 24]

# pose estimation
cfg.TRAINER.RANSAC_PIXEL_THR = 0.5

cfg.TRAINER.OPTIMIZER = "adamw"
cfg.TRAINER.ADAMW_DECAY = 0.1
cfg.TRAINER.NUM_WORKERS = 4  # replace num_workers for command line
cfg.TRAINER.ckpt_save_dir = './tmp_ckpt'  # replace num_workers for command line
cfg.TRAINER.dataset_sink_mode = False
cfg.LOFTR.MATCH_COARSE.TRAIN_COARSE_PERCENT = 0.3

# system config
cfg.system.mode = 0  # 0 for graph mode, 1 for pynative mode in MindSpore
cfg.system.distribute = False
cfg.system.device_id = 0
cfg.system.amp_level = 'O0'
cfg.system.drop_overflow_update = True
cfg.system.val_interval = 1
cfg.system.val_while_train = True

cfg.metrics.main_indicator = 'auc@5'
