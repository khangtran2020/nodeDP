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

    tr_idx = get_index_by_value(a=graph.ndata['train_mask'], val=1)
    te_idx = get_index_by_value(a=graph.ndata['test_mask'], val=1)

    with torch.no_grad():
        pred_full = model.full(g=graph, x=graph.ndata['feat']).detach()
        entr = -1*pred_full*torch.log(pred_full)
        print("Entropy:", entr)
        entr = entr.sum(dim=-1)
        tr_conf = entr[tr_idx]
        te_conf = entr[te_idx]

        top_conf_tr, tr_idx_conf = torch.topk(tr_conf, num_half, largest=False)
        top_conf_te, te_idx_conf = torch.topk(te_conf, num_half, largest=False)

        rprint(f"Top entropy train has size {top_conf_tr.size()}, test has size {top_conf_te.size()}")
        rprint(f"Mean entropy train {top_conf_tr.mean()}, mean entropy test {top_conf_te.mean()}")

        tr_idx = tr_idx[tr_idx_conf]
        te_idx = te_idx[te_idx_conf]

        num_tr = tr_idx.size(dim=0)
        num_te = te_idx.size(dim=0)
    
    perm = torch.randperm(num_tr, device=device)
    tr_idx = tr_idx[perm]

    perm = torch.randperm(num_te, device=device)
    te_idx = te_idx[perm]

    shadow_idx = torch.cat((tr_idx, te_idx), dim=0).to(device)

    graph.ndata['shadow_idx'] = torch.zeros(graph.nodes().size(dim=0)).to(device)
    graph.ndata['shadow_label'] = torch.zeros(graph.nodes().size(dim=0)).to(device)

    graph.ndata['shadow_idx'][shadow_idx] += 1
    graph.ndata['shadow_label'][tr_idx] += 1
    graph.ndata['shadow_label'][te_idx] += -1

    shadow_graph = graph.subgraph(shadow_idx)
    rprint(f'Shadow graph has: {shadow_graph.nodes().size(dim=0)} nodes, label counts: {shadow_graph.ndata["shadow_label"].unique(return_counts=True)}')

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
# F.one_hot(, num_classes=5)
    feature = torch.cat((feature, y_pred.detach()), dim=-1)
    feature = torch.cat((feature, F.one_hot(y_label.detach())), dim=-1)

    rprint(f"Shadow label: {shadow_graph.ndata['shadow_label'].unique(return_counts=True)}")
    # shadow_graph.ndata['shadow_label']
    id_pos = get_index_by_value(a = shadow_graph.ndata['shadow_label'], val=1)
    id_neg = get_index_by_value(a = shadow_graph.ndata['shadow_label'], val=-1)

    x_pos_mean = feature[id_pos].mean(dim=0)
    x_neg_mean = feature[id_neg].mean(dim=0)

    rprint(f"Difference in mean of the features: {(x_pos_mean - x_neg_mean).norm(p=2).item()}")
    shadow_graph.ndata['shadow_label'] = ((shadow_graph.ndata['shadow_label']+1)/2).int()
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