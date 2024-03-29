import torch
import torch.nn.functional as F
import numpy as np
from Utils.utils import get_index_by_value, get_index_by_not_list
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from rich import print as rprint

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

    feature = None        
    for i, los in enumerate(loss):
        pred = y_pred[i].detach().clone()
        label = y_label[i].detach().clone()
        grad = torch.Tensor([]).to(device)
        los.backward(retain_graph=True)
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad = torch.cat((grad, p.grad.detach().flatten()), dim = 0)
        feat = torch.cat((pred, torch.unsqueeze(label, dim=0), grad), dim = 0)
        feat = torch.unsqueeze(feat, dim = 0)
        if i == 0:
            feature = feat
        else:
            feature = torch.cat((feature, feat), dim=0)
        model.zero_grad()

    # rprint(f"Done generating feature for g_train: {x_tr_pos_feat.size()}, {x_te_pos_feat.size()}")

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