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

## Testing
```console
bash scripts/reproduce_test/outdoor_ds.sh 
```
