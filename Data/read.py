import os
import sys
import dgl
import pandas as pd
import torch
import numpy as np
import networkx as nx
from functools import partial
from Data.facebook import Facebook
from Data.amazon import Amazon
from copy import deepcopy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from Data.dataloader import NodeDataLoader
from ogb.nodeproppred import DglNodePropPredDataset
from Utils.utils import *
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from torch_geometric.transforms import Compose, RandomNodeSplit
from joblib import Parallel, delayed


def read_data(args, data_name, history):
    if data_name == 'reddit':
        data = dgl.data.RedditDataset()
        graph = data[0]
        node_split(graph=graph, val_size=0.1, test_size=0.15)
        list_of_label = filter_class_by_count(graph=graph, min_count=10000)
    elif data_name == 'cora':
        data = CoraGraphDataset()
        graph = data[0]
        node_split(graph=graph, val_size=0.1, test_size=0.15)
        list_of_label = filter_class_by_count(graph=graph, min_count=0)
    elif data_name == 'citeseer':
        data = CiteseerGraphDataset()
        graph = data[0]
        node_split(graph=graph, val_size=0.1, test_size=0.15)
        list_of_label = filter_class_by_count(graph=graph, min_count=0)
    elif data_name == 'pubmed':
        data = PubmedGraphDataset()
        graph = data[0]
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
    args.num_class = len(list_of_label)
    args.num_feat = graph.ndata['feat'].shape[1]
    graph = dgl.remove_self_loop(graph)
    if args.submode == 'density':
        if args.density < 1.0:
            graph = reduce_desity(g=graph, dens_reduction=args.density)
        else:
            graph = increase_density(args=args, g=graph, density_increase=args.density - 1)
    history['tr_id'] = get_index_bynot_value(a=graph.ndata['train_mask'], val=0)
    history['va_id'] = get_index_bynot_value(a=graph.ndata['val_mask'], val=0)
    history['te_id'] = get_index_bynot_value(a=graph.ndata['test_mask'], val=0)
    g_train, g_val, g_test = graph_split(graph=graph, drop=True)
    idx = torch.index_select(graph.nodes(), 0, graph.ndata['label_mask'].nonzero().squeeze()).numpy()
    graph = graph.subgraph(torch.LongTensor(idx))
    graph = drop_isolated_node(graph)
    args.num_data_point = len(g_train.nodes())
    return g_train, g_val, g_test, graph


def node_split(graph, val_size, test_size):
    keys = graph.ndata.keys()
    if 'train_mask' not in keys or 'val_mask' not in keys or 'test_mask' not in keys:
        node_id = np.arrange(len(graph.nodes()))
        node_label = graph.ndata['label'].numpy()
        id_train, id_test, y_train, y_test = train_test_split(node_id, node_label, test_size=test_size,
                                                              stratify=node_label)
        test_mask = np.zeros_like(node_id)
        test_mask[id_test] = 1
        id_train, id_val, y_train, y_val = train_test_split(id_train, y_train, test_size=val_size, stratify=y_train)
        train_mask = np.zeros_like(node_id)
        train_mask[id_train] = 1
        val_mask = np.zeros_like(node_id)
        val_mask[id_val] = 1
        graph.ndata['train_mask'] = torch.from_numpy(train_mask)
        graph.ndata['val_mask'] = torch.from_numpy(train_mask)
        graph.ndata['test_mask'] = torch.from_numpy(train_mask)


def graph_split(graph, drop=True):
    train_id = torch.index_select(graph.nodes(), 0, graph.ndata['train_mask'].nonzero().squeeze()).numpy()
    val_id = torch.index_select(graph.nodes(), 0, graph.ndata['val_mask'].nonzero().squeeze()).numpy()
    test_id = torch.index_select(graph.nodes(), 0, graph.ndata['test_mask'].nonzero().squeeze()).numpy()
    print(f"ORIGINAL GRAPH HAS: {graph.nodes().size()} nodes and {graph.edges()[0].size()} edges")
    train_g = graph.subgraph(torch.LongTensor(train_id))
    test_g = graph.subgraph(torch.LongTensor(test_id))
    val_g = graph.subgraph(torch.LongTensor(val_id))

    if drop == True:
        train_g = drop_isolated_node(train_g)
        val_g = drop_isolated_node(val_g)
        test_g = drop_isolated_node(test_g)

    return train_g, val_g, test_g


def drop_isolated_node(graph):
    mask = torch.zeros_like(graph.nodes())
    src, dst = graph.edges()
    mask[src.unique()] = 1
    mask[dst.unique()] = 1
    index = get_index_by_value(a=mask, val=1)
    nodes_id = graph.nodes()[index]
    return graph.subgraph(torch.LongTensor(nodes_id))


def filter_class_by_count(graph, min_count):
    target = deepcopy(graph.ndata['label'])
    print(target.unique(return_counts=True)[1])
    counts = target.unique(return_counts=True)[1] > min_count
    index = get_index_by_value(a=counts, val=True)
    label_dict = dict(zip(index.tolist(), range(len(index))))
    # print("Label Dict:", label_dict)
    mask = target.apply_(lambda x: x in index.tolist())
    graph.ndata['label'].apply_(lambda x: label_dict[x] if x in label_dict.keys() else -1)
    graph.ndata['train_mask'] = graph.ndata['train_mask'] & mask
    graph.ndata['val_mask'] = graph.ndata['val_mask'] & mask
    graph.ndata['test_mask'] = graph.ndata['test_mask'] & mask
    graph.ndata['label_mask'] = mask
    return index.tolist()


def init_loader(args, device, train_g, test_g, val_g):
    train_nodes = train_g.nodes()
    val_nodes = val_g.nodes()
    test_nodes = test_g.nodes()

    print('Nodes:', train_g.nodes().size(), val_g.nodes().size(), test_g.nodes().size())
    print('Edges:', train_g.edges()[0].size(), val_g.edges()[0].size(), test_g.edges()[0].size())
    print('Num label:', args.num_class)
    print('Test label dist:', test_g.ndata['label'].unique(return_counts=True))

    sampler = dgl.dataloading.NeighborSampler([args.n_neighbor for i in range(args.n_layers)])
    if args.mode == 'nodedp':
        train_loader = NodeDataLoader(g=train_g, batch_size=int(args.sampling_rate * len(train_nodes)),
                                      shuffle=True, num_workers=0,
                                      num_nodes=[args.n_neighbor for i in range(args.n_layers)], cache_result=False,
                                      device=device, sampling_rate=args.sampling_rate)
    else:
        train_loader = dgl.dataloading.DataLoader(train_g, train_nodes, sampler, device=device,
                                                  batch_size=args.batch_size, shuffle=True, drop_last=True,
                                                  num_workers=args.num_worker)
    val_loader = dgl.dataloading.DataLoader(val_g, val_nodes, sampler, device=device,
                                            batch_size=args.batch_size, shuffle=True, drop_last=False,
                                            num_workers=args.num_worker)
    test_loader = dgl.dataloading.DataLoader(test_g, test_nodes, sampler, device=device,
                                             batch_size=args.batch_size, shuffle=True, drop_last=False,
                                             num_workers=args.num_worker)
    return train_loader, val_loader, test_loader


def fold_separation(g, num_folds):
    skf = StratifiedKFold(n_splits=num_folds)
    node_id = range(len(g.nodes().tolist()))
    node_label = g.ndata['label'].tolist()
    folds = [x for x in skf.split(node_id, node_label)]
    return folds


def fold_assign(g, folds, current_fold):
    tr_mask = np.zeros(len(g.nodes().tolist()))
    va_mask = np.zeros(len(g.nodes().tolist()))
    train_id, valid_id = folds[current_fold]
    tr_mask[train_id] = 1
    va_mask[valid_id] = 1
    tr_mask = tr_mask.tolist()
    va_mask = va_mask.tolist()
    g.ndata['train_mask'] = torch.BoolTensor(tr_mask)
    g.ndata['val_mask'] = torch.BoolTensor(va_mask)
    return


def reduce_desity(g, dens_reduction):
    # num_edge = g.edges()[0].size(dim=0)
    # num_node = g.nodes().size(dim=0)
    src_edge, dst_edge = g.edges()
    index = (src_edge < dst_edge).nonzero(as_tuple=True)[0]
    src_edge = src_edge[index]
    dst_edge = dst_edge[index]

    num_edge = src_edge.size(dim=0)
    num_node = g.nodes().size(dim=0)

    dens = num_edge / num_node
    dens = dens * (1 - dens_reduction)
    num_edge_new = int(dens * num_node)
    indices = np.arange(num_edge)

    chosen_index = torch.from_numpy(np.random.choice(a=indices, size=num_edge_new, replace=False)).int()
    src_edge_new = torch.index_select(input=src_edge, dim=0, index=chosen_index)
    dst_edge_new = torch.index_select(input=dst_edge, dim=0, index=chosen_index)

    src_edge_undirected = torch.cat((src_edge_new, dst_edge_new), dim=0)
    dst_edge_undirected = torch.cat((dst_edge_new, src_edge_new), dim=0)

    new_g = dgl.graph((src_edge_undirected, dst_edge_undirected), num_nodes=num_node)
    new_g.ndata['feat'] = g.ndata['feat'].clone()
    new_g.ndata['label'] = g.ndata['label'].clone()
    new_g.ndata['train_mask'] = g.ndata['train_mask'].clone()
    new_g.ndata['val_mask'] = g.ndata['val_mask'].clone()
    new_g.ndata['test_mask'] = g.ndata['test_mask'].clone()
    new_g.ndata['label_mask'] = g.ndata['label_mask'].clone()
    new_g = drop_isolated_node(graph=new_g)
    print(f"Old # edges: {num_edge}, New # edges: {src_edge_new.size(dim=0)}")
    return new_g

def reduce_desity_deg(g, dens_reduction):
    # num_edge = g.edges()[0].size(dim=0)
    # num_node = g.nodes().size(dim=0)
    src_edge, dst_edge = g.edges()
    num_edge = src_edge.size(dim=0)
    num_node = g.nodes().size(dim=0)
    dens = num_edge / num_node
    dens = dens * (1 - dens_reduction)
    num_edge_new = int(dens * num_node)
    indices = np.arange(num_edge)
    chosen_index = torch.from_numpy(np.random.choice(a=indices, size=num_edge_new, replace=False)).int()
    src_edge_new = torch.index_select(input=src_edge, dim=0, index=chosen_index)
    dst_edge_new = torch.index_select(input=dst_edge, dim=0, index=chosen_index)
    new_g = dgl.graph((src_edge_new, dst_edge_new), num_nodes=num_node)
    new_g.ndata['feat'] = g.ndata['feat'].clone()
    new_g.ndata['label'] = g.ndata['label'].clone()
    new_g = drop_isolated_node(graph=new_g)
    print(f"Old # edges: {num_edge}, New # edges: {new_g.edges()[0].size(dim=0)}")
    return new_g


def read_data_attack(args, data_name, history):
    if data_name == 'reddit':
        data = dgl.data.RedditDataset()
        graph = data[0]
        node_split(graph=graph, val_size=0.1, test_size=0.15)
        list_of_label = filter_class_by_count(graph=graph, min_count=10000)
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
        list_of_label = filter_class_by_count(graph=graph, min_count=6000)
    args.num_class = len(list_of_label)
    args.num_feat = graph.ndata['feat'].shape[1]
    graph = dgl.remove_self_loop(graph)
    num_node = graph.ndata['feat'].size(dim=0)
    tr_mask = torch.zeros(size=num_node)
    va_mask = torch.zeros(size=num_node)
    te_mask = torch.zeros(size=num_node)
    tr_mask[history['tr_id']] = 1
    va_mask[history['va_id']] = 1
    te_mask[history['te_id']] = 1
    graph.ndata['train_mask'] = tr_mask
    graph.ndata['val_mask'] = va_mask
    graph.ndata['test_mask'] = te_mask
    g_train, g_val, g_test = graph_split(graph=graph, drop=True)
    if args.submode == 'density':
        g_train = reduce_desity(g=g_train, dens_reduction=args.density)
    args.num_data_point = len(g_train.nodes())
    return g_train, g_val, g_test, graph


def init_shadow_loader(args, device, graph):
    tr_nid = get_index_by_value(a=graph.ndata['str_mask'], val=1).to(device)
    va_nid = get_index_by_value(a=graph.ndata['sva_mask'], val=1).to(device)
    te_nid = get_index_by_value(a=graph.ndata['sva_mask'], val=1).to(device)

    sampler = dgl.dataloading.NeighborSampler([args.n_neighbor for i in range(args.n_layers)])
    tr_loader = dgl.dataloading.DataLoader(graph.to(device), tr_nid.to(device), sampler, device=device,
                                           batch_size=args.batch_size, shuffle=True, drop_last=True,
                                           num_workers=args.num_worker)

    va_loader = dgl.dataloading.DataLoader(graph.to(device), va_nid.to(device), sampler, device=device,
                                           batch_size=args.batch_size, shuffle=False, drop_last=False,
                                           num_workers=args.num_worker)

    te_loader = dgl.dataloading.DataLoader(graph.to(device), te_nid.to(device), sampler, device=device,
                                           batch_size=args.batch_size, shuffle=False, drop_last=False,
                                           num_workers=args.num_worker)
    return tr_loader, va_loader, te_loader


def randomsplit(graph, num_node_per_class, train_ratio=0.7, test_ratio=0.2):
    y = graph.ndata['label']
    num_classes = y.max().item() + 1
    num_train_nodes_per_class = int(num_node_per_class * train_ratio)
    num_test_nodes_per_class = int(num_node_per_class * test_ratio)

    train_mask = torch.zeros_like(y)
    test_mask = torch.zeros_like(y)
    val_mask = torch.zeros_like(y)

    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        num_nodes = idx.size(0)
        if num_nodes >= num_node_per_class:
            idx = idx[torch.randperm(idx.size(0))][:num_node_per_class]
            train_mask[idx[:num_train_nodes_per_class]] = True
            test_mask[idx[num_train_nodes_per_class:num_train_nodes_per_class + num_test_nodes_per_class]] = True
            val_mask[idx[num_train_nodes_per_class + num_test_nodes_per_class:]] = True

    graph.ndata['str_mask'] = train_mask
    graph.ndata['sva_mask'] = val_mask
    graph.ndata['ste_mask'] = test_mask


def read_pair_file(args, file_path='Data/wpair/', nodes=None):
    temp = None
    data_name = args.dataset if args.dataset != 'amazon' else 'arxiv'
    for i in range(1,6):
        print(f"Redding file: {data_name}_{i}_{data_name}_{i}-ranked.wpairs")
        file_name = f'{data_name}_{i}_{data_name}_{i}-ranked.wpairs'
        df = pd.read_csv(f'{file_path}{file_name}', delimiter='\t', header=None)
        if i == 1:
            temp = df.values
        else:
            temp = np.concatenate((temp, df.values), axis=0)
    temp = temp[np.where(np.isin(temp[:, 0], nodes) & np.isin(temp[:, 1], nodes))[0]]
    print(f"Number of new possible edges is: {temp.shape[0]}")
    temp = temp[temp[:, 2].argsort()]
    return temp

def increase_density(args, g, density_increase):

    if os.path.exists(f'Data/pairs/{args.dataset}.npy') == False:
        nodes = g.nodes()
        perm_indx = torch.randperm(n=nodes.size(dim=0))
        nodes = nodes[perm_indx]
        del perm_indx
        num_node = 2000
        num_batch = int(nodes.size(dim=0)/num_node) + 1
        adj = g.adj_external(scipy_fmt='csr')
        G = nx.from_scipy_sparse_matrix(adj)
        link_pred = partial(link_prediction_on_sub_graph, num_node=num_node, nodes=nodes, org_graph=g, org_graph_nx=G)
        results = Parallel(n_jobs=os.cpu_count(), prefer="threads")(delayed(link_pred)(i) for i in range(num_batch))
        res = [r for r in results]
        results_ = []
        for r in res: results_.extend(r)
        results_ = sorted(results_, key=lambda x: x[-1], reverse=True)
        results_ = np.array(results_).astype(int)
        np.save(f'Data/pairs/{args.dataset}.npy', results_)
        rprint(f"Saved file to directory: Data/pairs/{args.dataset}.npy")
    else:
        results_ = np.load(f'Data/pairs/{args.dataset}.npy')
        rprint(f"Loaded file from directory: Data/pairs/{args.dataset}.npy")
    
    src_edge, dst_edge = g.edges()
    index = (src_edge < dst_edge).nonzero(as_tuple=True)[0]
    src_edge = src_edge[index]
    dst_edge = dst_edge[index]

    num_edge = src_edge.size(dim=0)
    num_node = g.nodes().size(dim=0)
    num_edge_new = int(density_increase * num_edge)
    indices = np.arange(results_.shape[0])
    
    choosen_index = np.random.choice(a=indices, size=num_edge_new, replace=False)
    new_src_edge = torch.from_numpy(results_[choosen_index, 0]).int()
    new_dst_edge = torch.from_numpy(results_[choosen_index, 1]).int()

    # print(new_src_edge.size(), new_dst_edge.size())

    src_edge_undirected = torch.cat((src_edge, new_src_edge, dst_edge, new_dst_edge), dim=0)
    dst_edge_undirected = torch.cat((dst_edge, new_dst_edge, src_edge, new_src_edge), dim=0)

    new_g = dgl.graph((src_edge_undirected, dst_edge_undirected), num_nodes=num_node)
    new_g.ndata['feat'] = g.ndata['feat'].clone()
    new_g.ndata['label'] = g.ndata['label'].clone()
    new_g.ndata['train_mask'] = g.ndata['train_mask'].clone()
    new_g.ndata['val_mask'] = g.ndata['val_mask'].clone()
    new_g.ndata['test_mask'] = g.ndata['test_mask'].clone()
    new_g.ndata['label_mask'] = g.ndata['label_mask'].clone()
    new_g = drop_isolated_node(graph=new_g)
    print(f"Old # edges: {num_edge}, New # edges: {new_src_edge.size(dim=0) + num_edge}")
    return new_g

def link_prediction_on_sub_graph(indx, num_node, nodes, org_graph, org_graph_nx):
    rprint(f"Running process {indx}")
    sub_node = nodes[indx*num_node:(indx+1)*num_node]
    sub_graph = org_graph.subgraph(sub_node)
    adj = sub_graph.adj_external(scipy_fmt='csr').todense()
    iu1 = np.triu_indices(adj.shape[0], 1)
    index = np.where(adj[iu1] < 1)[1]
    src_edge, dst_edge = iu1
    src_edge = src_edge[index]
    dst_edge = dst_edge[index]
    # print(f"Start predicting for process {indx}")
    pair = list(zip(src_edge, dst_edge))
    preds = nx.jaccard_coefficient(org_graph_nx, pair)
    new_pair = []
    for u, v, p in tqdm(preds):
        if p > 0:
            new_pair.append([u, v, p])
    rprint(f"Done process {indx}, new pair has size {len(new_pair)}")
    return new_pair