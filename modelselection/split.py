import numpy as np


np.random.seed(42)
data = np.random.rand(100, 5)
targets = np.random.randint(0, 2, size=100)


def train_test_split(data, targets, test_size=0.2):
    num_samples = len(data)
    num_test_samples = int(test_size * num_samples)

    indices = np.random.permutation(num_samples)
    test_indices = indices[:num_test_samples]
    train_indices = indices[num_test_samples:]

    train_data = data[train_indices]
    train_targets = targets[train_indices]
    test_data = data[test_indices]
    test_targets = targets[test_indices]

    return train_data, test_data, train_targets, test_targets

train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.2)
