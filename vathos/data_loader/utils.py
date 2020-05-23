import torch


def split_dataset(dataset, train_ratio=0.7):
    # 70 - 30 split
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size

    train_subset, test_subset = torch.utils.data.random_split(
        dataset, [train_size, test_size])

    train_subset = torch.utils.data.Subset(
        train_subset, range(0, len(train_subset)//16))
    test_subset = torch.utils.data.Subset(
        train_subset, range(0, len(test_subset)//16))

    return train_subset, test_subset
