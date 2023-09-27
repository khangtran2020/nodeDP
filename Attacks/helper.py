import gc
import torch
import torch.nn.functional as F
import numpy as np
from Utils.utils import get_index_by_value, get_index_by_not_list
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from rich import print as rprint

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
    

def generate_attack_samples_white_box(graph, model, criter, device):
    
    num_tr = graph.ndata['train_mask'].sum().item()
    num_te = graph.ndata['test_mask'].sum().item()
    num_half = min(num_tr, num_te)

    with torch.no_grad():
        y_pred = model.full(g=graph, x=graph.ndata['feat'])
        entr = get_entropy(pred=y_pred)

    _, indx = torch.topk(entr, num_half, largest=False)

    tr_idx = get_index_by_value(a=graph.ndata['train_mask'][indx], val=1)
    perm = torch.randperm(tr_idx.size(dim=0), device=device)
    tr_idx = tr_idx[perm]


    _, indx = torch.topk(entr, num_half, largest=True)


    te_idx = get_index_by_value(a=graph.ndata['test_mask'][indx], val=1)
    perm = torch.randperm(te_idx.size(dim=0), device=device)
    te_idx = te_idx[perm]

    rprint(f'Train index has: {tr_idx.size(dim=0)} nodes, Test index has: {te_idx.size(dim=0)} nodes')


    shadow_idx = torch.cat((tr_idx, te_idx), dim=0).to(device)

    graph.ndata['shadow_idx'] = torch.zeros(graph.nodes().size(dim=0)).to(device)
    graph.ndata['shadow_label'] = torch.zeros(graph.nodes().size(dim=0)).to(device)

    graph.ndata['shadow_idx'][shadow_idx] += 1
    graph.ndata['shadow_label'][tr_idx] += 1
    graph.ndata['shadow_label'][te_idx] += -1

    shadow_graph = graph.subgraph(shadow_idx)


    sh_node_id = shadow_graph.nodes().tolist()
    sh_node_label = shadow_graph.ndata['shadow_label'].tolist()

    id_tr, id_te, y_tr, y_te = train_test_split(sh_node_id, sh_node_label, stratify=sh_node_label, test_size=0.2)
    id_va, id_va, y_tr, y_va = train_test_split(id_tr, y_tr, stratify=y_tr, test_size=0.16)

    shadow_graph.ndata['train_shadow_mask'] = torch.zeros(shadow_graph.nodes().size(dim=0)).to(device)
    shadow_graph.ndata['val_shadow_mask'] = torch.zeros(shadow_graph.nodes().size(dim=0)).to(device)
    shadow_graph.ndata['test_shadow_mask'] = torch.zeros(shadow_graph.nodes().size(dim=0)).to(device)

    shadow_graph.ndata['train_shadow_mask'][id_tr] += 1
    shadow_graph.ndata['val_shadow_mask'][id_va] += 1
    shadow_graph.ndata['test_shadow_mask'][id_te] += 1

    del y_pred
    gc.collect()

    y_pred = model.full(g=shadow_graph, x=shadow_graph.ndata['feat'])
    y_label = shadow_graph.ndata['label']
    loss = criter(y_pred, y_label)

    feature = torch.Tensor([]).to(device)
    for i, los in enumerate(loss):
        grad = torch.Tensor([]).to(device)
        los.backward(retain_graph=True)
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad = torch.cat((grad, p.grad.detach().flatten()), dim = 0)
        feature = torch.cat((feature, torch.unsqueeze(grad, dim = 0)), dim=0)
        model.zero_grad()

    feature = torch.cat((feature, y_pred.detach(), torch.unsqueeze(y_label.detach(), dim =0)), dim=1)
    rprint(f"Feature has size: {feature.size()}")
    shadow_graph.ndata['shadow_label'] = ((shadow_graph.ndata['shadow_label']+1)/2).int()
    rprint(f"Shadow label: {shadow_graph.ndata['shadow_label'].unique(return_counts=True)}")
    # shadow_graph.ndata['shadow_label']
    id_pos = get_index_by_value(a = shadow_graph.ndata['shadow_label'], val=1)
    id_neg = get_index_by_value(a = shadow_graph.ndata['shadow_label'], val=-1)

    x_pos_mean = feature[id_pos].mean(dim=0)
    x_neg_mean = feature[id_neg].mean(dim=0)

    rprint(f"Difference in mean of the features: {(x_pos_mean - x_neg_mean).norm(p=2).item()}")

    x_tr, y_tr = feature[id_tr], shadow_graph.ndata['shadow_label'][id_tr]
    x_va, y_va = feature[id_va], shadow_graph.ndata['shadow_label'][id_va]
    x_te, y_te = feature[id_te], shadow_graph.ndata['shadow_label'][id_te]

    return x_tr, x_va, x_te, y_tr, y_va, y_te


def generate_attack_samples_white_box_grad(graph, device):

    tr_mask = 'train_mask'
    te_mask = 'test_mask'
    num_train = graph.ndata[tr_mask].sum()
    num_test = graph.ndata[te_mask].sum()
    num_half = min(num_train, num_test)

    nodes_id = graph.nodes()

    perm = torch.randperm(num_train, device=device)[:num_half]
    idx = get_index_by_value(a=graph.ndata[tr_mask], val=1)
    idx_pos = nodes_id[idx][perm]

    perm = torch.randperm(num_test, device=device)[:num_half]
    idx = get_index_by_value(a=graph.ndata[te_mask], val=1)
    idx_neg = nodes_id[idx][perm]

    y_pos = torch.ones(num_half)
    y_neg = torch.zeros(num_half)

    return idx_pos, idx_neg, y_pos, y_neg



def get_entropy(pred):
    log_pred = torch.log2(pred)
    temp = -1*pred*log_pred
    return temp.sum(dim=-1)