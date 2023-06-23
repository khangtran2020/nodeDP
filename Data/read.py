import dgl
import torch
import numpy as np

from copy import deepcopy
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from Data.dataloader import NodeDataLoader
from ogb.nodeproppred import DglNodePropPredDataset
from Utils.utils import get_index_by_value

def read_data(args, data_name, ratio):
    if data_name == 'cora':
        data = dgl.data.CoraGraphDataset()
        graph = data[0]
        node_split(graph=graph, val_size=0.1, test_size=0.15)
        list_of_label = filter_class_by_count(graph=graph, min_count=1)
    elif data_name == 'citeseer':
        data = dgl.data.CiteseerGraphDataset()
        graph = data[0]
        node_split(graph=graph, val_size=0.1, test_size=0.15)
        list_of_label = filter_class_by_count(graph=graph, min_count=1)
    elif data_name == 'pubmed':
        data = dgl.data.PubmedGraphDataset()
        graph = data[0]
        node_split(graph=graph, val_size=0.1, test_size=0.15)
        list_of_label = filter_class_by_count(graph=graph, min_count=1)
    elif data_name == 'ogbn-arxiv':
        data = DglNodePropPredDataset('ogbn-arxiv')
        graph, node_labels = data[0]
        graph = dgl.add_reverse_edges(graph)
        graph.ndata['label'] = node_labels[:, 0]
        node_split(graph=graph, val_size=0.1, test_size=0.15)
        list_of_label = filter_class_by_count(graph=graph, min_count=1)
    elif data_name == 'reddit':
        data = dgl.data.RedditDataset()
        graph = data[0]
        node_split(graph=graph, val_size=0.1, test_size=0.15)
        list_of_label = filter_class_by_count(graph=graph, min_count = 10000)
    args.num_class = len(list_of_label)
    args.num_feat = graph.ndata['feat'].shape[1]
    graph = dgl.remove_self_loop(graph)
    g_train, g_val, g_test = graph_split(graph=graph)
    args.num_data_point = len(g_train.nodes())
    return g_train, g_val, g_test


def node_split(graph, val_size, test_size):
    keys = graph.ndata.keys()
    if 'train_mask' not in keys or 'val_mask' not in keys or 'test_mask' not in keys:
        node_id = np.arrange(len(graph.nodes()))
        node_label = graph.ndata['label'].numpy()
        id_train, id_test, y_train, y_test = train_test_split(node_id, node_label, test_size=test_size, stratify=node_label)
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


def graph_split(graph):

    train_id = torch.index_select(graph.nodes(), 0, graph.ndata['train_mask'].nonzero().squeeze()).numpy()
    val_id = torch.index_select(graph.nodes(), 0, graph.ndata['val_mask'].nonzero().squeeze()).numpy()
    test_id = torch.index_select(graph.nodes(), 0, graph.ndata['test_mask'].nonzero().squeeze()).numpy()

    train_g = graph.subgraph(torch.LongTensor(train_id))
    test_g = graph.subgraph(torch.LongTensor(test_id))
    val_g = graph.subgraph(torch.LongTensor(val_id))

    train_g = drop_isolated_node(train_g)
    val_g = drop_isolated_node(val_g)
    test_g = drop_isolated_node(test_g)

    return train_g, val_g, test_g

def drop_isolated_node(graph):
    mask = torch.zeros_like(graph.nodes())
    src, dst = graph.edges()
    mask[src] = 1
    mask[dst] = 1
    index = get_index_by_value(a=mask, val=1)
    nodes_id = graph.nodes()[index]
    return graph.subgraph(torch.LongTensor(nodes_id))

def filter_class_by_count(graph, min_count):
    target = deepcopy(graph.ndata['label'])
    counts = target.unique(return_counts=True)[1] > min_count
    index = get_index_by_value(a=counts, val=True)
    label_dict = dict(zip(index.tolist(), range(len(index))))
    # print("Label Dict:", label_dict)
    mask = target.apply_(lambda x: x in index.tolist())
    graph.ndata['label'].apply_(lambda x: label_dict[x] if x in label_dict.keys() else -1)
    graph.ndata['train_mask'] = graph.ndata['train_mask'] & mask
    graph.ndata['val_mask'] = graph.ndata['val_mask'] & mask
    graph.ndata['test_mask'] = graph.ndata['test_mask'] & mask
    return index.tolist()

def init_loader(args, device, train_g, test_g, val_g):

    train_nodes = train_g.nodes()
    val_nodes = val_g.nodes()
    test_nodes = test_g.nodes()

    # print('Nodes:', train_g.nodes().size(), val_g.nodes().size(), test_g.nodes().size())
    # print('Edges:', train_g.edges()[0].size(), val_g.edges()[0].size(), test_g.edges()[0].size())
    # print('Num label:', args.num_class)
    # print('Test label dist:', test_g.ndata['label'].unique(return_counts=True))

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
