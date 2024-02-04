# LoFTR-ms
LoFTR implemented with mindspore framework

## Preprocessing Dataset
1. Convert NPZ files for training:
```console
python scripts/mega_depth_idx_convert.py 
```

2. Convert image files for training:
```console
python scripts/mega_depth_preprocess.py 
```

3. Convert NPZ files for testing:
```console
python scripts/mega_depth_test_idx_convert.py 
```

4. Create link:
```console
sh scripts/mega_depth_link.sh 
```

## Training
```console
bash scripts/reproduce_train/outdoor_ds.sh 
```

## Distribute Training
Before performing distributed training, it is necessary to modify the [loftr configuration](./configs/loftr/outdoor/loftr_ds_dense.py) file as follows:
   ```diff
   - cfg.system.distribute = False
   + cfg.system.distribute = True
   ```
Then run the following command:
```console
bash scripts/reproduce_train/distribute_outdoor_ds.sh 
```

## Testing
```console
bash scripts/reproduce_test/outdoor_ds.sh 
```
