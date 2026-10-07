import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# No Normalize: pixels stay in [0,1], which makes Task 2 (DLG) and PSNR simpler.
_tf = transforms.ToTensor()


def load_mnist(root="./data"):
    train = datasets.MNIST(root, train=True, download=True, transform=_tf)
    test = datasets.MNIST(root, train=False, download=True, transform=_tf)
    return train, test


def split_clients(train_set, num_clients=4, samples_per_client=2000, seed=0):
    """IID split: each client gets a disjoint random subset of MNIST."""
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(train_set), generator=g).tolist()
    return [
        Subset(train_set, perm[i * samples_per_client:(i + 1) * samples_per_client])
        for i in range(num_clients)
    ]


def test_loader(test_set, batch_size=256):
    return DataLoader(test_set, batch_size=batch_size, shuffle=False)
