'''
This file is currently not used!
'''
import mindspore as ms
from mindspore import ops


class RandomConcatSampler:
    """ Random sampler for ConcatDataset. At each epoch, `n_samples_per_subset` samples will be draw from each subset
    in the ConcatDataset. If `subset_replacement` is ``True``, sampling within each subset will be done with replacement.
    However, it is impossible to sample data without replacement between epochs, unless bulding a stateful sampler lived along the entire training phase.
    
    For current implementation, the randomness of sampling is ensured no matter the sampler is recreated across epochs or not and call `torch.manual_seed()` or not.
    Args:
        shuffle (bool): shuffle the random sampled indices across all sub-datsets.
        repeat (int): repeatedly use the sampled indices multiple times for training.
            [arXiv:1902.05509, arXiv:1901.09335]
    NOTE: Don't re-initialize the sampler between epochs (will lead to repeated samples)
    NOTE: This sampler behaves differently with DistributedSampler.
          It assume the dataset is splitted across ranks instead of replicated.
    TODO: Add a `set_epoch()` method to fullfill sampling without replacement across epochs.
          ref: https://github.com/PyTorchLightning/pytorch-lightning/blob/e9846dd758cfb1500eb9dba2d86f6912eb487587/pytorch_lightning/trainer/training_loop.py#L373
    """
    def __init__(self,
                 data_source,
                 n_samples_per_subset,
                 subset_replacement=True,
                 shuffle=True,
                 repeat=1,
                 seed=None):
        self.data_source = data_source
        self.n_subset = len(self.data_source.datasets)
        self.n_samples_per_subset = n_samples_per_subset
        self.n_samples = self.n_subset * self.n_samples_per_subset * repeat
        self.subset_replacement = subset_replacement
        self.repeat = repeat
        self.shuffle = shuffle
        self.generator = seed
        
    def __len__(self):
        return self.n_samples
    
    def __iter__(self):
        indices = []
        # sample from each sub-dataset
        for d_idx in range(self.n_subset):
            low = 0 if d_idx==0 else self.data_source.cumulative_sizes[d_idx-1]
            high = self.data_source.cumulative_sizes[d_idx]
            if self.subset_replacement:
                rand_tensor = ops.randint(low, high, (self.n_samples_per_subset, ),
                                            seed=self.generator, dtype=ms.int32)
            else:  # sample without replacement
                len_subset = len(self.data_source.datasets[d_idx])
                rand_tensor = ops.randperm(len_subset) + low
                if len_subset >= self.n_samples_per_subset:
                    rand_tensor = rand_tensor[:self.n_samples_per_subset]
                else: # padding with replacement
                    rand_tensor_replacement = ops.randint(low, high, (self.n_samples_per_subset - len_subset, ),
                                                            seed=self.generator, dtype=ms.int32)
                    rand_tensor = ops.cat([rand_tensor, rand_tensor_replacement])
            indices.append(rand_tensor)
        indices = ops.cat(indices)
        if self.shuffle:  # shuffle the sampled dataset (from multiple subsets)
            rand_tensor = ops.randperm(len(indices), seed=self.generator)
            indices = indices[rand_tensor]

        # repeat the sampled indices (can be used for RepeatAugmentation or pure RepeatSampling)
        if self.repeat > 1:
            repeat_indices = [indices for _ in range(self.repeat - 1)]
            if self.shuffle:
                _choice = lambda x: x[ops.randperm(len(x), seed=self.generator)]
                repeat_indices = map(_choice, repeat_indices)
            indices = ops.cat([indices, *repeat_indices], 0)
        
        # assert indices.shape[0] == self.n_samples
        return iter(indices.asnumpy().tolist())
