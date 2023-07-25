import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset

class Data(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = self.X.size(dim=0)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len

def generate_attack_samples(graph, tar_conf, mode, device):

    tr_mask = 'train_mask' if mode == 'target' else 'str_mask'
    va_mask = 'val_mask' if mode == 'target' else 'sva_mask'
    te_mask = 'test_mask' if mode == 'target' else 'ste_mask'

    num_classes = tar_conf.size(1)
    print(num_classes, graph.ndata['label'].unique())
    num_train = graph.ndata[tr_mask].sum()
    num_test = graph.ndata[te_mask].sum()
    num_half = min(num_train, num_test)

    labels = F.one_hot(graph.ndata['label'], num_classes).float().to(device)
    samples = torch.cat([tar_conf, labels], dim=1).to(device)

    perm = torch.randperm(num_train, device=device)[:num_half]
    pos_samples = samples[graph.ndata[tr_mask]][perm]

    perm = torch.randperm(num_test, device=device)[:num_half]
    neg_samples = samples[graph.ndata[te_mask]][perm]

    # pos_entropy = Categorical(probs=pos_samples[:, :num_classes]).entropy().mean()
    # neg_entropy = Categorical(probs=neg_samples[:, :num_classes]).entropy().mean()

    # console.debug(f'pos_entropy: {pos_entropy:.4f}, neg_entropy: {neg_entropy:.4f}')

    x = torch.cat([neg_samples, pos_samples], dim=0)
    y = torch.cat([
        torch.zeros(num_half, dtype=torch.long, device=device),
        torch.ones(num_half, dtype=torch.long, device=device),
    ])

    # shuffle data
    perm = torch.randperm(2 * num_half, device=device)
    x, y = x[perm], y[perm]

    return x, y