import torch
import torch.nn.functional as F
import numpy as np
from Utils.utils import get_index_by_value, get_index_by_not_list
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
    

def generate_attack_samples_white_box(tr_g, te_g, model, criter, device):

    tr_mask = 'train_mask'
    te_mask = 'test_mask'

    num_tr = tr_g.ndata[tr_mask].sum()
    num_te = te_g.ndata[te_mask].sum()
    num_half = min(num_tr, num_te)

    num_tr_att = int(0.8 * num_half)
    num_te_att = num_half - num_tr_att

    te_node = te_g.nodes()
    perm = torch.randperm(num_te, device=device)[:num_half]
    idx = get_index_by_value(a=te_g.ndata[tr_mask], val=1)
    idx_te = te_node[idx][perm]

    idx_neg_tr = idx_te[:num_tr_att]
    idx_neg_te = idx_te[num_tr_att:]

    y_te_pred = model.full(g=te_g, x=te_g.ndata['feat'])
    y_te_label = te_g.ndata['label']
    loss_te = criter(y_te_pred, y_te_label)

    x_tr_neg_feat = None        
    for i, idx in enumerate(idx_neg_tr):
        pred = y_te_pred[idx].clone()
        label = y_te_label[idx].clone()
        grad = torch.Tensor([]).to(device)
        loss_te[idx].backward(retain_graph=True)
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad = torch.cat((grad, p.grad.detach().flatten()), dim = 0)
        feat = torch.cat((pred.detach(), torch.unsqueeze(label.detach(), dim=0), grad), dim = 0)
        feat = torch.unsqueeze(feat, dim = 0)
        if i == 0:
            x_tr_neg_feat = feat
        else:
            x_tr_neg_feat = torch.cat((x_tr_neg_feat, feat), dim=0)
        model.zero_grad()

    x_te_neg_feat = None        
    for i, idx in enumerate(idx_neg_te):
        pred = y_te_pred[idx].clone()
        label = y_te_label[idx].clone()
        grad = torch.Tensor([]).to(device)
        loss_te[idx].backward(retain_graph=True)
        for name, p in model.named_parameters():
            if p.grad is not None:
                grad = torch.cat((grad, p.grad.detach().flatten()), dim = 0)
        feat = torch.cat((pred.detach(), torch.unsqueeze(label.detach(), dim=0), grad), dim = 0)
        feat = torch.unsqueeze(feat, dim = 0)
        if i == 0:
            x_te_neg_feat = feat
        else:
            x_te_neg_feat = torch.cat((x_te_neg_feat, feat), dim=0)
        model.zero_grad()

    tr_node = tr_g.nodes()
    perm = torch.randperm(num_tr, device=device)[:num_tr_att]
    idx = get_index_by_value(a=tr_g.ndata[tr_mask], val=1)
    idx_tr = tr_node[idx][perm]

    g_sh = tr_g.subgraph(idx_tr)
    y_sh_pred = model.full(g=g_sh, x=g_sh.ndata['feat'])
    y_sh_label = g_sh.ndata['label']
    loss_sh = criter(y_sh_pred, y_sh_label)

    x_tr_pos_feat = None        
    for i, idx in enumerate(g_sh.nodes()):
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
            x_tr_pos_feat = feat
        else:
            x_tr_pos_feat = torch.cat((x_tr_pos_feat, feat), dim=0)
        model.zero_grad()

    idx_left = get_index_by_not_list(arr=tr_g.nodes(), test_arr=idx_tr)
    g_un = tr_g.subgraph(idx_left)
    perm = torch.randperm(g_un.nodes().size(dim=0), device=device)[:num_te_att]
    idx_te_neg = g_un.nodes()[perm]

    
    y_un_pred = model.full(g=g_un, x=g_un.ndata['feat'])
    y_un_label = g_un.ndata['label']
    loss_un = criter(y_un_pred, y_un_label)

    x_te_pos_feat = None        
    for i, idx in enumerate(idx_te_neg):
        pred = y_un_pred[idx].clone()
        label = y_un_label[idx].clone()
        grad = torch.Tensor([]).to(device)
        loss_un[idx].backward(retain_graph=True)
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