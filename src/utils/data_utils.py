def get_data_subset(dataset, subset_size, seed):
    """
    Get a subset of a Hugging Face Dataset or DatasetDict using either
    an absolute number of samples or a ratio.

    Args:
        subset_size (int | float): number of samples or ratio (0 < ratio <= 1)
        dataset: a Dataset or DatasetDict
        seed (int): random seed for shuffling

    Returns:
        Dataset or DatasetDict: subset of the input dataset
    """
    from datasets import DatasetDict

    def _subset_split(ds_split):
        # Determine number of samples
        if isinstance(subset_size, float):
            if not (0 < subset_size <= 1):
                raise ValueError("Ratio must be between 0 and 1.")
            n_samples = int(len(ds_split) * subset_size)
        else:
            n_samples = subset_size

        if n_samples == -1 or n_samples >= len(ds_split):
            return ds_split
        return ds_split.shuffle(seed=seed).select(range(n_samples))

    # Handle DatasetDict
    if isinstance(dataset, DatasetDict):
        return DatasetDict({split: _subset_split(ds_split) for split, ds_split in dataset.items()})
    else:
        return _subset_split(dataset)