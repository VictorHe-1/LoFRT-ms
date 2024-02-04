import numpy as np
import bisect


class RandomConcatSampler:
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
        self.generator = np.random.default_rng(seed)

    def __len__(self):
        return self.n_samples

    def __iter__(self):
        indices = []
        # sample from each sub-dataset
        for d_idx in range(self.n_subset):
            low = 0 if d_idx == 0 else self.data_source.cumulative_sizes[d_idx - 1]
            high = self.data_source.cumulative_sizes[d_idx]
            if self.subset_replacement:
                rand_tensor = self.generator.choice(np.arange(low, high), size=self.n_samples_per_subset, replace=True)
            else:  # sample without replacement
                len_subset = len(self.data_source.datasets[d_idx])
                rand_tensor = self.generator.permutation(len_subset) + low
                if len_subset >= self.n_samples_per_subset:
                    rand_tensor = rand_tensor[:self.n_samples_per_subset]
                else:  # padding with replacement
                    rand_tensor_replacement = self.generator.choice(np.arange(low, high),
                                                                    size=self.n_samples_per_subset - len_subset,
                                                                    replace=True)
                    rand_tensor = np.concatenate((rand_tensor, rand_tensor_replacement))
            indices.append(rand_tensor)
        indices = np.concatenate(indices)
        if self.shuffle:  # shuffle the sampled dataset (from multiple subsets)
            rand_tensor = self.generator.permutation(len(indices))
            indices = indices[rand_tensor]

        # repeat the sampled indices (can be used for RepeatAugmentation or pure RepeatSampling)
        if self.repeat > 1:
            repeat_indices = [indices.copy() for _ in range(self.repeat - 1)]
            if self.shuffle:
                for repeat_index in repeat_indices:
                    self.generator.shuffle(repeat_index)
            indices = np.concatenate([indices, *repeat_indices], axis=0)

        assert indices.shape[0] == self.n_samples
        return iter(indices.tolist())


if __name__ == "__main__":
    class ConcatDataset:
        @staticmethod
        def cumsum(sequence):
            r, s = [], 0
            for e in sequence:
                l = len(e)
                r.append(l + s)
                s += l
            return r

        def __init__(self, datasets):
            if len(datasets) == 0:
                raise ValueError("datasets passed to ConcatDataset cannot be empty!")
            self.datasets = datasets
            self.cumulative_sizes = self.cumsum(self.datasets)
            self.sampler = None

        def get_output_columns(self):
            return self.output_columns

        def __len__(self):
            return self.cumulative_sizes[-1]

        def __getitem__(self, idx):
            if idx < 0:
                if -idx > len(self):
                    raise ValueError("absolute value of index should not exceed dataset length")
                idx = len(self) + idx
            dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
            if dataset_idx == 0:
                sample_idx = idx
            else:
                sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
            return self.datasets[dataset_idx][sample_idx]


    # 创建两个数据集示例
    dataset1 = [1, 2, 3, 4]
    dataset2 = [6, 7, 8, 9, 10]

    # 创建ConcatDataset，将两个数据集合并在一起
    concat_dataset = ConcatDataset([dataset1, dataset2])

    # 创建RandomConcatSampler实例
    sampler = RandomConcatSampler(concat_dataset, n_samples_per_subset=2, subset_replacement=True, shuffle=True,
                                  repeat=1, seed=42)
    for value in sampler:
        print(value)
    print("-------------------------------")
    for value in sampler:
        print(value)
