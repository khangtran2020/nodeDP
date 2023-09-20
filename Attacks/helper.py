import torch
import torch.nn.functional as F
import numpy as np
from Utils.utils import get_index_by_value
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

def generate_attack_samples(tr_graph, tr_conf, mode, device, te_graph=None, te_conf=None):

    tr_mask = 'train_mask' if mode == 'target' else 'str_mask'
    te_mask = 'test_mask' if mode == 'target' else 'ste_mask'

    if mode != 'target':
        num_classes = tr_conf.size(1)
        print(num_classes, tr_graph.ndata['label'].unique())
        num_train = tr_graph.ndata[tr_mask].sum()
        num_test = tr_graph.ndata[te_mask].sum()
        num_half = min(num_train, num_test)

        labels = F.one_hot(tr_graph.ndata['label'], num_classes).float().to(device)
        samples = torch.cat([tr_conf, labels], dim=1).to(device)

        perm = torch.randperm(num_train, device=device)[:num_half]
        idx = get_index_by_value(a=tr_graph.ndata[tr_mask], val=1)
        pos_samples = samples[idx][perm]

        perm = torch.randperm(num_test, device=device)[:num_half]
        idx = get_index_by_value(a=tr_graph.ndata[te_mask], val=1)
        neg_samples = samples[idx][perm]

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

    else:
        num_classes = tr_conf.size(1)
        print(num_classes, tr_graph.ndata['label'].unique())
        num_train = tr_graph.ndata[tr_mask].sum()
        num_test = te_graph.ndata[te_mask].sum()
        num_half = min(num_train, num_test)

        labels = F.one_hot(tr_graph.ndata['label'], num_classes).float().to(device)
        tr_samples = torch.cat([tr_conf, labels], dim=1).to(device)

        labels = F.one_hot(te_graph.ndata['label'], num_classes).float().to(device)
        te_samples = torch.cat([te_conf, labels], dim=1).to(device)

        # samples = torch.cat((tr_samples, te_samples), dim=0).to(device)

        perm = torch.randperm(num_train, device=device)[:num_half]
        idx = get_index_by_value(a=tr_graph.ndata[tr_mask], val=1)
        pos_samples = tr_samples[idx][perm]

        perm = torch.randperm(num_test, device=device)[:num_half]
        idx = get_index_by_value(a=te_graph.ndata[te_mask], val=1)
        neg_samples = te_samples[idx][perm]

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
    

def generate_attack_samples_white_box(graph, ratio, device):

    tr_mask = 'train_mask'
    te_mask = 'test_mask'
    num_train = graph.ndata[tr_mask].sum()
    num_test = graph.ndata[te_mask].sum()
    num_half = min(num_train, num_test)

    nodes_id = graph.nodes()

    perm = torch.randperm(num_train, device=device)[:num_half]
    idx = get_index_by_value(a=graph.ndata[tr_mask], val=1)
    idx_tr = nodes_id[idx][perm]

    perm = torch.randperm(num_test, device=device)[:num_half]
    idx = get_index_by_value(a=graph.ndata[te_mask], val=1)
    idx_te = nodes_id[idx][perm]

    num_tr = int(0.8 * idx_te.size(dim=0))
    num_te = idx_te.dim(dim=0) - num_tr

    x_tr = torch.cat((idx_tr[:num_tr], idx_te[:num_tr]), dim=0)
    y_tr = torch.cat((torch.ones(num_tr), torch.zeros(num_tr)), dim=0)
    x_te = torch.cat((idx_tr[num_tr:], idx_te[num_tr:]), dim=0)
    y_te = torch.cat((torch.ones(num_te), torch.zeros(num_te)), dim=0)

    perm = torch.randperm(x_tr.size(dim=0), device=device)
    x_tr = x_tr[perm]
    y_tr = y_tr[perm]

    perm = torch.randperm(x_te.size(dim=0), device=device)
    x_te = x_te[perm]
    y_te = y_te[perm]

    return x_tr, x_te, y_tr, y_te