import sys
import dgl
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.transforms import Compose, RandomNodeSplit
from Utils.utils import get_index_by_value, get_index_bynot_value, get_index_by_list
from sklearn.linear_model import LogisticRegression
from rich import print as rprint
from functools import partial
from dgl.dataloading import transforms
from dgl.dataloading.base import NID
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from Data.read import node_split, filter_class_by_count, graph_split, drop_isolated_node, reduce_desity
from Data.facebook import Facebook
from Data.amazon import Amazon


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
    rprint(f"Orginal graph: {graph}")

    if exist == False:

        tr_org_idx = get_index_by_value(a=graph.ndata['train_mask'], val=1)
        te_org_idx = get_index_by_value(a=graph.ndata['test_mask'], val=1)

        te_node = org_nodes[te_org_idx]
        tr_node = org_nodes[tr_org_idx]

        num_shadow = int(ratio * tr_node.size(dim=0))
        perm = torch.randperm(tr_node.size(dim=0))
        shatr_nodes = tr_node[perm[:num_shadow]]

        num_half = min(int(te_node.size(dim=0)*0.2), int(shatr_nodes.size(dim=0)*0.2))
        # print("Half", num_half)

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
        # pos_id = get_index_by_value(a=shadow_graph.ndata['pos_mask'], val=1)
        # neg_id = get_index_by_value(a=shadow_graph.ndata['neg_mask'], val=1)
        # rprint(f"Uniqueness of id_intr: {shadow_graph.ndata['id_intr'][pos_id].unique(return_counts=True)}")
        # rprint(f"Uniqueness of id_inte: {shadow_graph.ndata['id_inte'][neg_id].unique(return_counts=True)}")
        # pos_mask = shadow_graph.ndata['pos_mask'].clone()
        # train_mask = shadow_graph.ndata['train_mask'].clone()
        # neg_mask = shadow_graph.ndata['neg_mask'].clone()
        # test_mask = shadow_graph.ndata['test_mask'].clone()
        # rprint(f"Positive mask unique: {pos_mask.unique(return_counts=True)}, Train mask in shadow graph {train_mask.unique(return_counts=True)}")
        # rprint(f"Positive mask unique: {neg_mask.unique(return_counts=True)}, Train mask in shadow graph {test_mask.unique(return_counts=True)}")
        rprint(f"Shadow graph average node degree: {shadow_graph.in_degrees().sum() / (len(shadow_graph.in_degrees()) + 1e-12)}")
        per = partial(percentage_pos, graph=shadow_graph)
        percentage = []
        for node in shadow_graph.nodes():
            percentage.append(per(node))
        percentage = torch.Tensor(percentage)
        rprint(f"Shadow graph average percentage neighbor is pos: {percentage.sum().item() / (len(percentage) + 1e-12)}, with histogram {np.histogram(percentage.tolist(), bins=5)}")
        rprint(f"Shadow graph average percentage neighbor is neg: {1 - percentage.sum().item() / (len(percentage) + 1e-12)}")
        temp_pos = percentage*shadow_graph.ndata['pos_mask']
        temp_neg = percentage*shadow_graph.ndata['neg_mask']
        rprint(f"Shadow graph average percentage neighbor is pos of pos: {temp_pos.mean().item() / (len(temp_pos) + 1e-12)}")
        rprint(f"Shadow graph average percentage neighbor is pos of neg: {temp_neg.mean().item() / (len(temp_neg) + 1e-12)}")
    return shadow_graph

def percentage_pos(node, graph):
    frontier = graph.sample_neighbors(node, -1)
    mask = torch.zeros_like(frontier.nodes())
    src, dst = frontier.edges()
    mask[src.unique().long()] = 1
    mask[dst.unique().long()] = 1
    index = get_index_by_value(a=mask, val=1)
    nodes_id = frontier.nodes()[index]
    num_pos = graph.ndata['pos_mask'][nodes_id.long()].sum()
    num_neg = graph.ndata['neg_mask'][nodes_id.long()].sum()
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

def get_graph(data_name):

    if data_name == 'reddit':
        data = dgl.data.RedditDataset()
        graph = data[0]
        node_split(graph=graph, val_size=0.1, test_size=0.15)
        list_of_label = filter_class_by_count(graph=graph, min_count=10000)
    elif data_name == 'cora':
        data = CoraGraphDataset()
        graph = data[0]
        del(graph.ndata['train_mask'])
        del(graph.ndata['val_mask'])
        del(graph.ndata['test_mask'])
        node_split(graph=graph, val_size=0.1, test_size=0.15)
        list_of_label = filter_class_by_count(graph=graph, min_count=0)
    elif data_name == 'citeseer':
        data = CiteseerGraphDataset()
        graph = data[0]
        del(graph.ndata['train_mask'])
        del(graph.ndata['val_mask'])
        del(graph.ndata['test_mask'])
        node_split(graph=graph, val_size=0.1, test_size=0.15)
        list_of_label = filter_class_by_count(graph=graph, min_count=0)
    elif data_name == 'pubmed':
        data = PubmedGraphDataset()
        graph = data[0]
        del(graph.ndata['train_mask'])
        del(graph.ndata['val_mask'])
        del(graph.ndata['test_mask'])
        node_split(graph=graph, val_size=0.1, test_size=0.15)
        list_of_label = filter_class_by_count(graph=graph, min_count=0)
    elif data_name == 'facebook':
        load_data = partial(Facebook, name='UIllinois20', target='year',
                            transform=Compose([
                                RandomNodeSplit(num_val=0.1, num_test=0.15)
                                # FilterClassByCount(min_count=1000, remove_unlabeled=True)
                            ])
                            )
        data = load_data(root='dataset/')[0]
        src_edge = data.edge_index[0]
        dst_edge = data.edge_index[1]
        graph = dgl.graph((src_edge, dst_edge), num_nodes=data.x.size(dim=0))
        graph.ndata['feat'] = data.x
        graph.ndata['label'] = data.y
        graph.ndata['train_mask'] = data.train_mask
        graph.ndata['val_mask'] = data.val_mask
        graph.ndata['test_mask'] = data.test_mask
        list_of_label = filter_class_by_count(graph=graph, min_count=1000)
        # sys.exit()
    elif data_name == 'amazon':
        load_data = partial(Amazon,
                            transform=Compose([
                                RandomNodeSplit(num_val=0.1, num_test=0.15)
                            ])
                            )
        data = load_data(root='dataset/')[0]
        src_edge = data.edge_index[0]
        dst_edge = data.edge_index[1]
        graph = dgl.graph((src_edge, dst_edge), num_nodes=data.x.size(dim=0))
        graph.ndata['feat'] = data.x
        graph.ndata['label'] = data.y
        graph.ndata['train_mask'] = data.train_mask
        graph.ndata['val_mask'] = data.val_mask
        graph.ndata['test_mask'] = data.test_mask
        list_of_label = filter_class_by_count(graph=graph, min_count=6000)

    return graph, list_of_label

def read_data(args, history, exist=False):

    graph, list_of_label = get_graph(data_name=args.dataset)
    args.num_class = len(list_of_label)
    args.num_feat = graph.ndata['feat'].shape[1]
    graph = dgl.remove_self_loop(graph)
    graph.ndata['org_id'] = graph.nodes().clone()

    if exist == False:
        rprint(f"History is {exist} to exist, need to reinitialize")
        history['tr_id'] = graph.ndata['train_mask'].tolist()
        history['va_id'] = graph.ndata['val_mask'].tolist()
        history['te_id'] = graph.ndata['test_mask'].tolist()
    else:
        rprint(f"History is {exist} to exist, assigning masks according to previous run")
        del(graph.ndata['train_mask'])
        del(graph.ndata['val_mask'])
        del(graph.ndata['test_mask'])

        id_train = history['tr_id']
        id_val = history['va_id']
        id_test = history['te_id']

        graph.ndata['train_mask'] = torch.LongTensor(id_train)
        graph.ndata['val_mask'] = torch.LongTensor(id_val)
        graph.ndata['test_mask'] = torch.LongTensor(id_test)
    
    if args.submode == 'density':
        graph = reduce_desity(g=graph, dens_reduction=args.density)

    if (args.submode == 'density') and (args.density == 1.0):
        g_train, g_val, g_test = graph_split(graph=graph, drop=False)
    else:
        g_train, g_val, g_test = graph_split(graph=graph, drop=True)

    rprint(f"Training graph average node degree: {g_train.in_degrees().sum() / (len(g_train.in_degrees()) + 1e-12)}")
    rprint(f"Valid graph average node degree: {g_val.in_degrees().sum() / (len(g_val.in_degrees()) + 1e-12)}")
    rprint(f"Testing graph average node degree: {g_test.in_degrees().sum() / (len(g_test.in_degrees()) + 1e-12)}")



    train_mask = torch.zeros(graph.nodes().size(dim=0))
    val_mask = torch.zeros(graph.nodes().size(dim=0))
    test_mask = torch.zeros(graph.nodes().size(dim=0))

    id_intr = g_train.ndata['org_id']
    id_inva = g_val.ndata['org_id']
    id_inte = g_test.ndata['org_id']

    train_mask[id_intr] = 1
    val_mask[id_inva] = 1
    test_mask[id_inte] = 1

    graph.ndata['train_mask'] = train_mask
    graph.ndata['test_mask'] = val_mask
    graph.ndata['test_mask'] = test_mask

    graph.ndata['id_intr'] = (torch.zeros(graph.nodes().size(dim=0)) - 1).long()
    graph.ndata['id_intr'][id_intr] = g_train.nodes().clone().long()

    graph.ndata['id_inte'] = (torch.zeros(graph.nodes().size(dim=0)) - 1).long()
    graph.ndata['id_inte'][id_inte] = g_test.nodes().clone().long()

    idx = torch.index_select(graph.nodes(), 0, graph.ndata['label_mask'].nonzero().squeeze()).numpy()
    graph = graph.subgraph(torch.LongTensor(idx))
    if (args.submode == 'density') and (args.density != 1.0):
        graph = drop_isolated_node(graph)
    args.num_data_point = len(g_train.nodes())

    return g_train, g_val, g_test, graph

def shadow_split_whitebox_extreme(graph, ratio, history=None, exist=False, diag=False):

    org_nodes = graph.nodes()
    tr_org_idx = get_index_by_value(a=graph.ndata['train_mask'], val=1)
    te_org_idx = get_index_by_value(a=graph.ndata['test_mask'], val=1)
    rprint(f"Orginal graph: {graph}")

    if exist == False:

        tr_org_idx = get_index_by_value(a=graph.ndata['train_mask'], val=1)
        te_org_idx = get_index_by_value(a=graph.ndata['test_mask'], val=1)

        te_node = org_nodes[te_org_idx]
        tr_node = org_nodes[tr_org_idx]

        num_shadow = int(ratio * tr_node.size(dim=0))
        perm = torch.randperm(tr_node.size(dim=0))
        shatr_nodes = tr_node[perm[:num_shadow]]

        num_half = min(int(te_node.size(dim=0)*0.4), int(shatr_nodes.size(dim=0)*0.4))
        # print("Half", num_half)

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
    
    temp_shadow_graph = graph.subgraph(shadow_nodes)

    pos_mask = temp_shadow_graph.ndata['pos_mask'].clone()
    src_edge, dst_edge = temp_shadow_graph.edges()
    src_edge_pos = pos_mask[src_edge].int().clone()
    dst_edge_pos = pos_mask[dst_edge].int().clone()
    same_pos = torch.logical_not(torch.logical_xor(src_edge_pos, dst_edge_pos)).int()
    indx = get_index_bynot_value(a=same_pos,val=0)
    src_edge = src_edge[indx].clone()
    dst_edge = dst_edge[indx].clone()

    neg_mask = temp_shadow_graph.ndata['neg_mask'].clone()
    src_edge_neg = neg_mask[src_edge].int().clone()
    dst_edge_neg = neg_mask[dst_edge].int().clone()
    same_neg = torch.logical_not(torch.logical_xor(src_edge_neg, dst_edge_neg)).int()
    indx = get_index_bynot_value(a=same_neg,val=0)
    src_edge = src_edge[indx].clone()
    dst_edge = dst_edge[indx].clone()

    shadow_graph = dgl.graph((src_edge, dst_edge), num_nodes=temp_shadow_graph.nodes().size(dim=0))
    for key in temp_shadow_graph.ndata.keys():
        shadow_graph.ndata[key] = temp_shadow_graph.ndata[key].clone()
    del temp_shadow_graph
    del src_edge
    del dst_edge
    del pos_mask
    del neg_mask

    if diag:
        rprint(f"Shadow graph average node degree: {shadow_graph.in_degrees().sum() / (len(shadow_graph.in_degrees()) + 1e-12)}")
        per = partial(percentage_pos, graph=shadow_graph)
        percentage = []
        for node in shadow_graph.nodes():
            percentage.append(per(node))
        percentage = torch.Tensor(percentage)
        rprint(f"Shadow graph average percentage neighbor is pos: {percentage.sum().item() / (len(percentage) + 1e-12)}, with histogram {np.histogram(percentage.tolist(), bins=5)}")
        rprint(f"Shadow graph average percentage neighbor is neg: {1 - percentage.sum().item() / (len(percentage) + 1e-12)}")
        temp_pos = percentage*shadow_graph.ndata['pos_mask']
        temp_neg = percentage*shadow_graph.ndata['neg_mask']
        rprint(f"Shadow graph average percentage neighbor is pos of pos: {temp_pos.mean().item() / (len(temp_pos) + 1e-12)}")
        rprint(f"Shadow graph average percentage neighbor is pos of neg: {temp_neg.mean().item() / (len(temp_neg) + 1e-12)}")
    return shadow_graph

def sample_blocks(graph, nodes, n_layer, max_nei=2):
    blocks = []
    seed_nodes = nodes
    for i in reversed(range(n_layer)):
        frontier = graph.sample_neighbors(seed_nodes, max_nei)
        block = transforms.to_block(frontier, seed_nodes, include_dst_in_src=True)
        seed_nodes = block.srcdata[NID]
        blocks.insert(0, block)
        
    return blocks

def shadow_split_whitebox_subgraph(graph, tr_graph, te_graph, n_layer, max_nei, 
                                   ratio, history=None, exist=False, diag=False):

    org_nodes = graph.nodes()
    tr_org_idx = get_index_by_value(a=graph.ndata['train_mask'], val=1)
    te_org_idx = get_index_by_value(a=graph.ndata['test_mask'], val=1)
    rprint(f"Orginal graph: {graph}")
    num_train = tr_graph.nodes().size(dim=0)
    num_test = te_graph.nodes().size(dim=0)

    if exist == False:

        tr_org_idx = get_index_by_value(a=graph.ndata['train_mask'], val=1)
        te_org_idx = get_index_by_value(a=graph.ndata['test_mask'], val=1)

        te_node = org_nodes[te_org_idx]
        tr_node = org_nodes[tr_org_idx]

        num_shadow = min(int(ratio * num_train), num_test)

        perm = torch.randperm(tr_node.size(dim=0))
        shapos_nodes = tr_node[perm[:num_shadow]]

        perm = torch.randperm(te_node.size(dim=0))
        shaneg_nodes = te_node[perm[:num_shadow]]

        num_half = min(int(shapos_nodes.size(dim=0)*0.4), int(shaneg_nodes.size(dim=0)*0.4))

        id_intr = graph.ndata['id_intr'][shapos_nodes]
        id_inte = graph.ndata['id_inte'][shaneg_nodes]
        block_pos = sample_blocks(graph=tr_graph, nodes=id_intr, n_layer=n_layer, max_nei=max_nei)
        block_neg = sample_blocks(graph=te_graph, nodes=id_inte, n_layer=n_layer, max_nei=max_nei)

        src_edge = torch.Tensor([])
        dst_edge = torch.Tensor([])
        sha_pos_nodes = torch.Tensor([])
        sha_neg_nodes = torch.Tensor([])

        for i in range(n_layer):

            src_pos = block_pos[i].srcdata['org_id']
            dst_pos = block_pos[i].dstdata['org_id']
            sha_pos_nodes = torch.cat((sha_pos_nodes, src_pos, dst_pos), dim=0)

            src_pos_edge, dst_pos_edge = block_pos[i].edges()
            src_pos_edge = src_pos[src_pos_edge]
            dst_pos_edge = dst_pos[dst_pos_edge]

            src_edge = torch.cat((src_edge, src_pos_edge), dim=0)
            dst_edge = torch.cat((dst_edge, dst_pos_edge), dim=0)


            src_neg = block_neg[i].srcdata['org_id']
            dst_neg = block_neg[i].dstdata['org_id']
            sha_neg_nodes = torch.cat((sha_neg_nodes, src_neg, dst_neg), dim=0)

            src_neg_edge, dst_neg_edge = block_neg[i].edges()
            src_neg_edge = src_neg[src_neg_edge]
            dst_neg_edge = dst_neg[dst_neg_edge]

            src_edge = torch.cat((src_edge, src_neg_edge), dim=0)
            dst_edge = torch.cat((dst_edge, dst_neg_edge), dim=0)

        src_edge = src_edge.int()
        dst_edge = dst_edge.int()

        g = dgl.graph((src_edge, dst_edge), num_nodes=graph.nodes().size(dim=0))
        for key in graph.ndata.keys():
            g.ndata[key] = graph.ndata[key].clone()

        g = dgl.to_simple(g, return_counts='cnt', writeback_mapping=False)
        sha_pos_nodes = sha_pos_nodes.unique()
        sha_neg_nodes = sha_neg_nodes.unique()
        shadow_nodes = torch.cat((sha_pos_nodes, sha_neg_nodes), dim=0).int()


        perm = torch.randperm(sha_pos_nodes.size(dim=0))
        sha_pos_te = sha_pos_nodes[perm[:num_half]]
        sha_pos_tr = sha_pos_nodes[perm[num_half:]]

        perm = torch.randperm(sha_neg_nodes.size(dim=0))
        sha_neg_te = sha_neg_nodes[perm[:num_half]]
        sha_neg_tr = sha_neg_nodes[perm[num_half:]]

        rprint(f"Shadow positive nodes to train: {sha_pos_tr.size(dim=0)}, to test: {sha_pos_te.size(dim=0)}")
        rprint(f"Shadow negative nodes to train: {sha_neg_tr.size(dim=0)}, to test: {sha_neg_te.size(dim=0)}")

        g_nodes = g.nodes()
        train_mask = torch.zeros(g_nodes.size(dim=0))
        test_mask = torch.zeros(g_nodes.size(dim=0))

        pos_mask_tr = torch.zeros(g_nodes.size(dim=0))
        pos_mask_te = torch.zeros(g_nodes.size(dim=0))

        neg_mask_tr = torch.zeros(g_nodes.size(dim=0))
        neg_mask_te = torch.zeros(g_nodes.size(dim=0))
        
        pos_mask = torch.zeros(g_nodes.size(dim=0))
        neg_mask = torch.zeros(g_nodes.size(dim=0))

        membership_label = torch.zeros(g_nodes.size(dim=0))

        sha_pos_tr = sha_pos_tr.long()
        sha_neg_tr = sha_neg_tr.long()
        sha_pos_te = sha_pos_te.long()
        sha_neg_te = sha_neg_te.long()

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

        src_edge, dst_edge = g.edges()
        for key in graph.ndata.keys():
            g.ndata[key] = graph.ndata[key].clone()


        history['sha_tr'] = train_mask.tolist()
        history['sha_src_edge'] = src_edge.tolist()
        history['sha_dst_edge'] = dst_edge.tolist()
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
        src_edge = torch.LongTensor(history['sha_src_edge'])
        dst_edge = torch.LongTensor(history['sha_dst_edge'])

        g = dgl.graph((src_edge, dst_edge), num_nodes=graph.nodes().size(dim=0))
        for key in graph.ndata.keys():
            g.ndata[key] = graph.ndata[key].clone()

        g.ndata['str_mask'] = train_mask
        g.ndata['ste_mask'] = test_mask
        g.ndata['sha_label'] = torch.Tensor(history['sha_label'])
        g.ndata['pos_mask'] = pos_mask
        g.ndata['neg_mask'] = neg_mask
        g.ndata['pos_mask_tr'] = pos_mask_tr
        g.ndata['pos_mask_te'] = pos_mask_te
        g.ndata['neg_mask_tr'] = neg_mask_tr
        g.ndata['neg_mask_te'] = neg_mask_te
    
    shadow_graph = g.subgraph(shadow_nodes)

    if diag:
        rprint(f"Shadow graph average node degree: {shadow_graph.in_degrees().sum() / (len(shadow_graph.in_degrees()) + 1e-12)}")
        per = partial(percentage_pos, graph=shadow_graph)
        percentage = []
        for node in shadow_graph.nodes():
            percentage.append(per(node))
        percentage = torch.Tensor(percentage)
        rprint(f"Shadow graph average percentage neighbor is pos: {percentage.sum().item() / (len(percentage) + 1e-12)}, with histogram {np.histogram(percentage.tolist(), bins=5)}")
        rprint(f"Shadow graph average percentage neighbor is neg: {1 - percentage.sum().item() / (len(percentage) + 1e-12)}")
        temp_pos = percentage*shadow_graph.ndata['pos_mask']
        temp_neg = percentage*shadow_graph.ndata['neg_mask']
        rprint(f"Shadow graph average percentage neighbor is pos of pos: {temp_pos.mean().item() / (len(temp_pos) + 1e-12)}")
        rprint(f"Shadow graph average percentage neighbor is pos of neg: {temp_neg.mean().item() / (len(temp_neg) + 1e-12)}")
    return shadow_graph

def shadow_split_whitebox_drop(graph, ratio, history=None, exist=False, diag=False):

    org_nodes = graph.nodes()
    rprint(f"Orginal graph: {graph}")

    if exist == False:

        tr_org_idx = get_index_by_value(a=graph.ndata['train_mask'], val=1)
        te_org_idx = get_index_by_value(a=graph.ndata['test_mask'], val=1)

        num_train = tr_org_idx.size(dim=0)
        num_test = te_org_idx.size(dim=0)

        te_node = org_nodes[te_org_idx]
        tr_node = org_nodes[tr_org_idx]

        num_shadow = min(int(ratio * num_train), num_test)

        perm = torch.randperm(tr_node.size(dim=0))
        shatr_nodes = tr_node[perm[:num_shadow]]
        
        num_half = min(int(te_node.size(dim=0)*0.2), int(shatr_nodes.size(dim=0)*0.2))

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
    
    temp_shgraph = graph.subgraph(shadow_nodes)
    src_edge, dst_edge = temp_shgraph.edges()
    edge_weight = torch.zeros(src_edge.size(dim=0))

    tr_pos_mask = temp_shgraph.ndata['pos_mask_tr']
    tr_neg_mask = temp_shgraph.ndata['neg_mask_tr']

    src_edge_inpos = tr_pos_mask[src_edge]
    dst_edge_inpos = tr_pos_mask[dst_edge]
    pos_intr_edges = torch.logical_and(src_edge_inpos, dst_edge_inpos).int()
    index = get_index_bynot_value(a=pos_intr_edges, val=0)
    edge_weight[index] = 1

    src_edge_inneg = tr_neg_mask[src_edge]
    dst_edge_inneg = tr_neg_mask[dst_edge]
    neg_intr_edges = torch.logical_and(src_edge_inneg, dst_edge_inneg).int()
    index = get_index_bynot_value(a=neg_intr_edges, val=0)
    edge_weight[index] = 1

    te_pos_mask = temp_shgraph.ndata['pos_mask_te']
    te_neg_mask = temp_shgraph.ndata['neg_mask_te']

    src_edge_inpos_te = te_pos_mask[src_edge]
    dst_edge_inpos_te = te_pos_mask[dst_edge]
    pos_inte_edges = torch.logical_or(src_edge_inpos_te, dst_edge_inpos_te).int()
    index = get_index_bynot_value(a=pos_inte_edges, val=0)
    edge_weight[index] = 0.5

    src_edge_inneg_te = te_neg_mask[src_edge]
    dst_edge_inneg_te = te_neg_mask[dst_edge]
    neg_inte_edges = torch.logical_or(src_edge_inneg_te, dst_edge_inneg_te).int()
    index = get_index_bynot_value(a=neg_inte_edges, val=0)
    edge_weight[index] = 0.5

    index_keep = get_index_bynot_value(a=edge_weight, val=0)
    src_edge = src_edge[index_keep]
    dst_edge = dst_edge[index_keep]
    edge_weight = edge_weight[index_keep]
    rprint(f"Distribution of edge_weight: {edge_weight.unique(return_counts=True)}")

    shadow_graph = dgl.graph((src_edge, dst_edge), num_nodes=temp_shgraph.nodes().size(dim=0))
    for key in temp_shgraph.ndata.keys():
        shadow_graph.ndata[key] = temp_shgraph.ndata[key].clone()

    shadow_graph.edata['prob'] = edge_weight

    if diag:
        rprint(f"Shadow graph average node degree: {shadow_graph.in_degrees().sum() / (len(shadow_graph.in_degrees()) + 1e-12)}")
        per = partial(percentage_pos, graph=shadow_graph)
        percentage = []
        for node in shadow_graph.nodes():
            percentage.append(per(node))
        percentage = torch.Tensor(percentage)
        rprint(f"Shadow graph average percentage neighbor is pos: {percentage.sum().item() / (len(percentage) + 1e-12)}, with histogram {np.histogram(percentage.tolist(), bins=5)}")
        rprint(f"Shadow graph average percentage neighbor is neg: {1 - percentage.sum().item() / (len(percentage) + 1e-12)}")
        temp_pos = percentage*shadow_graph.ndata['pos_mask']
        temp_neg = percentage*shadow_graph.ndata['neg_mask']
        rprint(f"Shadow graph average percentage neighbor is pos of pos: {temp_pos.mean().item() / (len(temp_pos) + 1e-12)}")
        rprint(f"Shadow graph average percentage neighbor is pos of neg: {temp_neg.mean().item() / (len(temp_neg) + 1e-12)}")
    return shadow_graph