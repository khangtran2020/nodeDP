import dgl
import torch
import numpy as np
from torch.utils.data import Dataset
from Utils.utils import get_index_by_value
from sklearn.linear_model import LogisticRegression
from rich import print as rprint
from functools import partial


class Data(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.len = self.X.size(dim=0)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

    def __len__(self):
        return self.len

def test_distribution_shift(x_tr, x_te):
    num_data = x_tr.size(dim=0) + x_te.size(dim=0)
    perm = torch.randperm(num_data)
    x = torch.cat((x_tr, x_te), dim=0).cpu()[perm]
    for i in range(x.size(dim=1)):
        x[:, i] = (x[:,i] - x[:,i].mean()) / (x[:,i].std() + 1e-12)
    x = x.numpy()
    y = torch.cat((torch.ones(x_tr.size(dim=0)), torch.zeros(x_te.size(dim=0))), dim=0).cpu()[perm].numpy()
    lr = LogisticRegression()
    lr.fit(X=x, y=y)
    rprint(f"Distribution shift accuracy: {lr.score(X=x, y=y)}")

def shadow_split(graph, ratio, train_ratio=0.6, test_ratio=0.4, history=None, exist=False):

    if exist == False:
        y = graph.ndata['label']
        num_classes = y.max().item() + 1

        train_mask = torch.zeros_like(y)
        test_mask = torch.zeros_like(y)

        for c in range(num_classes):
            idx = (y == c).nonzero(as_tuple=False).view(-1)
            num_nodes = idx.size(0)
            num_shadow = int(ratio*num_nodes)
            num_tr = int(train_ratio*num_shadow)
            num_te = int(test_ratio*num_shadow)

            idx = idx[torch.randperm(idx.size(0))][:num_shadow]
            train_mask[idx[:num_tr]] = True
            test_mask[idx[num_tr:]] = True

        graph.ndata['str_mask'] = train_mask
        graph.ndata['ste_mask'] = test_mask

        tr_idx = get_index_by_value(a=train_mask, val=1).tolist()
        te_idx = get_index_by_value(a=test_mask, val=1).tolist()
        history['sha_tr'] = tr_idx
        history['sha_te'] = te_idx
    
    else:    
        train_mask = torch.zeros(graph.nodes().size(dim=0))
        test_mask = torch.zeros(graph.nodes().size(dim=0))

        train_mask[history['sha_tr']] = 1
        test_mask[history['sha_te']] = 1

        graph.ndata['str_mask'] = train_mask
        graph.ndata['ste_mask'] = test_mask

def shadow_split_whitebox(graph, ratio, history=None, exist=False, diag=False):

    org_nodes = graph.nodes()
    tr_org_idx = get_index_by_value(a=graph.ndata['train_mask'], val=1)
    te_org_idx = get_index_by_value(a=graph.ndata['test_mask'], val=1)
    # rprint(f"test orginal nodes: {te_org_idx}")

    if exist == False:

        tr_org_idx = get_index_by_value(a=graph.ndata['train_mask'], val=1)
        te_org_idx = get_index_by_value(a=graph.ndata['test_mask'], val=1)

        te_node = org_nodes[te_org_idx]
        tr_node = org_nodes[tr_org_idx]

        num_shadow = int(ratio * tr_node.size(dim=0))
        perm = torch.randperm(tr_node.size(dim=0))
        shatr_nodes = tr_node[perm[:num_shadow]]

        num_half = min(int(te_node.size(dim=0) / 2), int(shatr_nodes.size(dim=0) / 2))

        perm = torch.randperm(shatr_nodes.size(dim=0))
        sha_pos_te = shatr_nodes[perm[:num_half]]
        sha_pos_tr = shatr_nodes[perm[num_half:]]

        perm = torch.randperm(te_node.size(dim=0))
        sha_neg_te = te_node[perm[:num_half]]
        sha_neg_tr = te_node[perm[num_half:]]

        rprint(f"Shadow positive nodes to train: {sha_pos_tr.size(dim=0)}, to test: {sha_pos_te.size(dim=0)}")
        rprint(f"Shadow negative nodes to train: {sha_neg_tr.size(dim=0)}, to test: {sha_neg_te.size(dim=0)}")

        train_mask = torch.zeros(org_nodes.size(dim=0))
        test_mask = torch.zeros(org_nodes.size(dim=0))

        pos_mask_tr = torch.zeros(org_nodes.size(dim=0))
        pos_mask_te = torch.zeros(org_nodes.size(dim=0))

        neg_mask_tr = torch.zeros(org_nodes.size(dim=0))
        neg_mask_te = torch.zeros(org_nodes.size(dim=0))
        
        pos_mask = torch.zeros(org_nodes.size(dim=0))
        neg_mask = torch.zeros(org_nodes.size(dim=0))

        membership_label = torch.zeros(org_nodes.size(dim=0))

        train_mask[sha_pos_tr] = 1
        train_mask[sha_neg_tr] = 1

        test_mask[sha_pos_te] = 1
        test_mask[sha_neg_te] = 1

        pos_mask_tr[sha_pos_tr] = 1
        pos_mask_te[sha_pos_te] = 1

        neg_mask_tr[sha_neg_tr] = 1
        neg_mask_te[sha_neg_te] = 1

        pos_mask[sha_pos_tr] = 1
        pos_mask[sha_pos_te] = 1

        neg_mask[sha_neg_tr] = 1
        neg_mask[sha_neg_te] = 1

        membership_label[sha_pos_tr] = 1
        membership_label[sha_pos_te] = 1

        membership_label[sha_neg_tr] = -1
        membership_label[sha_neg_te] = -1

        graph.ndata['str_mask'] = train_mask
        graph.ndata['ste_mask'] = test_mask
        graph.ndata['sha_label'] = membership_label
        graph.ndata['pos_mask'] = pos_mask
        graph.ndata['neg_mask'] = neg_mask
        graph.ndata['pos_mask_tr'] = pos_mask_tr
        graph.ndata['pos_mask_te'] = pos_mask_te
        graph.ndata['neg_mask_tr'] = neg_mask_tr
        graph.ndata['neg_mask_te'] = neg_mask_te

        shadow_nodes = torch.cat((shatr_nodes, te_node), dim=0)

        history['sha_tr'] = train_mask.tolist()
        history['sha_te'] = test_mask.tolist()
        history['sha_label'] = membership_label.tolist()
        history['shadow_nodes'] = shadow_nodes.tolist()
        history['pos_mask'] = pos_mask.tolist()
        history['neg_mask'] = neg_mask.tolist()
        history['pos_mask_tr'] = pos_mask_tr.tolist()
        history['pos_mask_te'] = pos_mask_te.tolist()
        history['neg_mask_tr'] = neg_mask_tr.tolist()
        history['neg_mask_te'] = neg_mask_te.tolist()

    
    else:    

        train_mask = torch.LongTensor(history['sha_tr'])
        test_mask = torch.LongTensor(history['sha_te'])
        shadow_nodes = torch.LongTensor(history['shadow_nodes'])
        pos_mask = torch.LongTensor(history['pos_mask'])
        neg_mask = torch.LongTensor(history['neg_mask'])
        pos_mask_tr = torch.LongTensor(history['pos_mask_tr'])
        pos_mask_te = torch.LongTensor(history['pos_mask_te'])
        neg_mask_tr = torch.LongTensor(history['neg_mask_tr'])
        neg_mask_te = torch.LongTensor(history['neg_mask_te'])

        graph.ndata['str_mask'] = train_mask
        graph.ndata['ste_mask'] = test_mask
        graph.ndata['sha_label'] = torch.Tensor(history['sha_label'])
        graph.ndata['pos_mask'] = pos_mask
        graph.ndata['neg_mask'] = neg_mask
        graph.ndata['pos_mask_tr'] = pos_mask_tr
        graph.ndata['pos_mask_te'] = pos_mask_te
        graph.ndata['neg_mask_tr'] = neg_mask_tr
        graph.ndata['neg_mask_te'] = neg_mask_te
    
    shadow_graph = graph.subgraph(shadow_nodes)
    if diag:
        rprint(f"Shadow graph average node degree: {shadow_graph.in_degrees().float().mean().item()}")
        per = partial(percentage_pos, graph=shadow_graph)
        percentage = []
        for node in shadow_graph.nodes():
            percentage.append(per(node))
        percentage = torch.Tensor(percentage)
        rprint(f"Shadow graph average percentage neighbor is pos: {percentage.mean().item()}, with histogram {np.histogram(percentage.tolist(), bins=5)}")
        rprint(f"Shadow graph average percentage neighbor is neg: {1 - percentage.mean().item()}")
        rprint(f"Shadow graph average percentage neighbor is pos of pos: {(percentage*shadow_graph.ndata['pos_mask']).mean().item()}")
        rprint(f"Shadow graph average percentage neighbor is pos of neg: {(percentage*shadow_graph.ndata['neg_mask']).mean().item()}")
    return shadow_graph

def percentage_pos(node, graph):
    frontier = graph.sample_neighbors(node, -1)
    mask = torch.zeros_like(frontier.nodes())
    src, dst = frontier.edges()
    mask[src.unique()] = 1
    mask[dst.unique()] = 1
    index = get_index_by_value(a=mask, val=1)
    nodes_id = frontier.nodes()[index]
    num_pos = graph.ndata['pos_mask'][nodes_id].sum()
    num_neg = graph.ndata['neg_mask'][nodes_id].sum()
    pos_percentage = num_pos.item() / (num_pos.item() + num_neg.item() + 1e-12)
    return pos_percentage


def init_shadow_loader(args, device, graph):

    tr_nid = get_index_by_value(a=graph.ndata['str_mask'], val=1).to(device)
    te_nid = get_index_by_value(a=graph.ndata['ste_mask'], val=1).to(device)

    sampler = dgl.dataloading.NeighborSampler([args.n_neighbor for i in range(args.n_layers)])
    tr_loader = dgl.dataloading.DataLoader(graph.to(device), tr_nid.to(device), sampler, device=device,
                                           batch_size=args.batch_size, shuffle=True, drop_last=True,
                                           num_workers=0)

    te_loader = dgl.dataloading.DataLoader(graph.to(device), te_nid.to(device), sampler, device=device,
                                           batch_size=args.batch_size, shuffle=False, drop_last=False,
                                           num_workers=0)
    return tr_loader, te_loader

def generate_attack_samples(graph, conf, nohop_conf, mode, device, te_graph=None, te_conf=None, te_nohop_conf=None):

    tr_mask = 'train_mask' if mode == 'target' else 'str_mask'
    te_mask = 'test_mask' if mode == 'target' else 'ste_mask'

    if mode != 'target':

        num_classes = conf.size(1)
        print(num_classes, graph.ndata['label'].unique())
        num_train = graph.ndata[tr_mask].sum()
        num_test = graph.ndata[te_mask].sum()
        num_half = min(num_train, num_test)

        # labels = F.one_hot(tr_graph.ndata['label'], num_classes).float().to(device)
        # top_k_conf, _ = torch.topk(conf, k=2)
        # top_k_nohop, _ = torch.topk(nohop_conf, k=2)
        samples = torch.cat([conf, nohop_conf], dim=1).to(device)

        perm = torch.randperm(num_train, device=device)[:num_half]
        idx = get_index_by_value(a=graph.ndata[tr_mask], val=1)
        pos_samples = samples[idx][perm]

        perm = torch.randperm(num_test, device=device)[:num_half]
        idx = get_index_by_value(a=graph.ndata[te_mask], val=1)
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
        num_classes = conf.size(1)
        print(num_classes, graph.ndata['label'].unique())
        num_train = graph.ndata[tr_mask].sum()
        num_test = te_graph.ndata[te_mask].sum()
        num_half = min(num_train, num_test)

        # labels = F.one_hot(tr_graph.ndata['label'], num_classes).float().to(device)
        # top_k_conf, _ = torch.topk(conf, k=2)
        # top_k_nohop, _ = torch.topk(nohop_conf, k=2)
        tr_samples = torch.cat([conf, nohop_conf], dim=1).to(device)

        # top_k_conf, _ = torch.topk(te_conf, k=2)
        # top_k_nohop, _ = torch.topk(te_nohop_conf, k=2)
        te_samples = torch.cat([te_conf, te_nohop_conf], dim=1).to(device)

        # samples = torch.cat((tr_samples, te_samples), dim=0).to(device)

        perm = torch.randperm(num_train, device=device)[:num_half]
        idx = get_index_by_value(a=graph.ndata[tr_mask], val=1)
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

def generate_nohop_graph(graph, device):

    nodes = graph.nodes().tolist()
    g = dgl.graph((nodes, nodes), num_nodes=len(nodes)).to(device)
    for key in graph.ndata.keys():
        g.ndata[key] = graph.ndata[key].clone()
    
    return g