import torch
import torch.nn.functional as F
import numpy as np
from Utils.utils import get_index_by_value, get_index_by_not_list
from torch.utils.data import Dataset
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
    perm = torch.randperm(num_tr, device=device)[:num_half]
    tr_idx = tr_idx[perm]

    te_idx = get_index_by_value(a=graph.ndata['test_mask'], val=1)
    perm = torch.randperm(num_te, device=device)[:num_half]
    te_idx = te_idx[perm]

    shadow_idx = torch.cat((tr_idx, te_idx), dim=0)
    shadow_label = torch.cat((torch.ones(num_half), torch.zeros(num_half)), dim=0)

    graph.ndata['shadow_idx'] = torch.zeros(graph.nodes().size(dim=0))
    graph.ndata['shadow_label'] = torch.zeros(graph.nodes().size(dim=0))

    graph.ndata['shadow_idx'][shadow_idx] += 1
    graph.ndata['shadow_label'][tr_idx] += 1
    graph.ndata['shadow_label'][te_idx] += -1

    shadow_graph = graph.subgraph(shadow_idx)
    rprint(f'Shadow graph has: {shadow_graph.nodes().size(dim=0)} nodes, label counts: {shadow_graph.ndata["shadow_label"].unique(return_counts=True)}')














    num_half = min(num_tr, num_te)

    num_tr_att = int(0.7 * num_half)
    num_te_att = num_half - num_tr_att

    te_node = te_g.nodes()
    perm = torch.randperm(num_te, device=device)
    idx_te = te_node[perm]

    idx_neg_tr = idx_te[:num_tr_att]
    idx_neg_te = idx_te[num_tr_att:]

    y_te_pred = model.full(g=te_g, x=te_g.ndata['feat'])
    y_te_label = te_g.ndata['label']
    loss_te = criter(y_te_pred, y_te_label)

    x_tr_neg_feat = None        
    for i, idx in enumerate(idx_neg_tr):
        pred = y_te_pred[idx].detach().clone()
        label = y_te_label[idx].detach().clone()
        grad = torch.Tensor([]).to(device)
        loss_te[idx].backward(retain_graph=True)
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad = torch.cat((grad, p.grad.detach().flatten()), dim = 0)
        feat = torch.cat((pred, torch.unsqueeze(label, dim=0), grad), dim = 0)
        feat = torch.unsqueeze(feat, dim = 0)
        if i == 0:
            x_tr_neg_feat = feat
        else:
            x_tr_neg_feat = torch.cat((x_tr_neg_feat, feat), dim=0)
        model.zero_grad()

    x_te_neg_feat = None        
    for i, idx in enumerate(idx_neg_te):
        pred = y_te_pred[idx].detach().clone()
        label = y_te_label[idx].detach().clone()
        grad = torch.Tensor([]).to(device)
        loss_te[idx].backward(retain_graph=True)
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad = torch.cat((grad, p.grad.detach().flatten()), dim = 0)
        feat = torch.cat((pred, torch.unsqueeze(label, dim=0), grad), dim = 0)
        feat = torch.unsqueeze(feat, dim = 0)
        if i == 0:
            x_te_neg_feat = feat
        else:
            x_te_neg_feat = torch.cat((x_te_neg_feat, feat), dim=0)
        model.zero_grad()

    rprint(f"Done generating feature for g_test: {x_tr_neg_feat.size()}, {x_te_neg_feat.size()}")    

    tr_node = tr_g.nodes()
    perm = torch.randperm(num_tr, device=device)
    idx_tr = tr_node[perm[:num_half]]

    g_sh = tr_g.subgraph(idx_tr)
    y_sh_pred = model.full(g=g_sh, x=g_sh.ndata['feat'])
    y_sh_label = g_sh.ndata['label']
    loss_sh = criter(y_sh_pred, y_sh_label)

    id_sh = g_sh.nodes()
    perm = torch.randperm(id_sh.size(dim=0), device=device) 
    id_sh_tr = id_sh[perm[:num_tr_att]]
    id_sh_te = id_sh[perm[num_tr_att:]]

    x_tr_pos_feat = None        
    for i, idx in enumerate(id_sh_tr):
        pred = y_sh_pred[idx].detach().clone()
        label = y_sh_label[idx].detach().clone()
        grad = torch.Tensor([]).to(device)
        loss_sh[idx].backward(retain_graph=True)
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad = torch.cat((grad, p.grad.detach().flatten()), dim = 0)
        feat = torch.cat((pred, torch.unsqueeze(label, dim=0), grad), dim = 0)
        feat = torch.unsqueeze(feat, dim = 0)
        if i == 0:
            x_tr_pos_feat = feat
        else:
            x_tr_pos_feat = torch.cat((x_tr_pos_feat, feat), dim=0)
        model.zero_grad()

    x_te_pos_feat = None        
    for i, idx in enumerate(id_sh_te):
        pred = y_sh_pred[idx].clone()
        label = y_sh_label[idx].clone()
        grad = torch.Tensor([]).to(device)
        loss_sh[idx].backward(retain_graph=True)
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad = torch.cat((grad, p.grad.detach().flatten()), dim = 0)
        feat = torch.cat((pred.detach(), torch.unsqueeze(label.detach(), dim=0), grad), dim = 0)
        feat = torch.unsqueeze(feat, dim = 0)
        if i == 0:
            x_te_pos_feat = feat
        else:
            x_te_pos_feat = torch.cat((x_te_pos_feat, feat), dim=0)
        model.zero_grad()

    rprint(f"Done generating feature for g_train: {x_tr_pos_feat.size()}, {x_te_pos_feat.size()}")

    x_pos_mean = torch.cat((x_tr_pos_feat, x_te_pos_feat), dim=0).mean(dim=0)
    x_neg_mean = torch.cat((x_tr_neg_feat, x_te_neg_feat), dim=0).mean(dim=0)

    rprint(f"Difference in mean of the features: {(x_pos_mean - x_neg_mean).norm(p=2).item()}")


    x_tr = torch.cat((x_tr_pos_feat, x_tr_neg_feat), dim=0)
    y_tr = torch.cat((torch.ones(x_tr_pos_feat.size(dim=0)), torch.zeros(x_tr_neg_feat.size(dim=0))), dim=0)
    perm = torch.randperm(x_tr.size(dim=0), device=device)
    x_tr = x_tr[perm]
    y_tr = y_tr[perm]


    x_te = torch.cat((x_te_pos_feat, x_te_neg_feat), dim=0)
    y_te = torch.cat((torch.ones(x_te_pos_feat.size(dim=0)), torch.zeros(x_te_neg_feat.size(dim=0))), dim=0)
    perm = torch.randperm(x_te.size(dim=0), device=device)
    x_te = x_te[perm]
    y_te = y_te[perm]

    return x_tr, x_te, y_tr, y_te


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