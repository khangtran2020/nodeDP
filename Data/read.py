import dgl
import torch
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from Data.dataloader import NodeDataLoader


def read_data(args, data_name, ratio):
    if data_name == 'cora':
        data = dgl.data.CoraGraphDataset()
    elif data_name == 'citeseer':
        data = dgl.data.CiteseerGraphDataset()
    elif data_name == 'pubmed':
        data = dgl.data.PubmedGraphDataset()
    args.num_feat = data[0].ndata['feat'].shape[1]
    args.num_class = data.num_classes
    g_train, g_test, folds = graph_split(data=data, ratio=ratio, folds=args.folds)
    return g_train, g_test, folds


def graph_split(data, ratio, folds):
    g = data[0]
    node_id = g.nodes().numpy()
    node_label = g.ndata['label'].numpy()
    print(node_id.shape, node_label.shape)
    id_train, id_test, y_train, y_test = train_test_split(node_id, node_label, test_size=ratio, stratify=node_label)
    train_g = g.subgraph(torch.LongTensor(id_train))
    test_g = g.subgraph(torch.LongTensor(id_test))
    folds = fold_separation(train_g, num_folds=folds)
    return train_g, test_g, folds


def init_loader(args, device, train_g, test_g, num_fold, fold):

    fold_assign(g=train_g, folds=num_fold, current_fold=fold)
    train_nodes = torch.index_select(train_g.nodes(), 0, train_g.ndata['train_mask'].nonzero().squeeze())
    val_nodes = torch.index_select(train_g.nodes(), 0, train_g.ndata['val_mask'].nonzero().squeeze())

    val_g = train_g.subgraph(torch.LongTensor(val_nodes))
    train_g = train_g.subgraph(torch.LongTensor(train_nodes))
    sampler = dgl.dataloading.NeighborSampler([args.n_neighbor for i in range(args.n_hid)])
    if args.mode == 'dp':
        train_loader = NodeDataLoader(g=train_g, batch_size=args.batch_size, shuffle=False, num_workers=0,
                                      num_nodes=[args.n_neighbor for i in range(args.n_hid)], cache_result=False,
                                      device=device)
    else:
        train_loader = dgl.dataloading.DataLoader(train_g, train_g.nodes(), sampler, device=device,
                                                  batch_size=args.batch_size, shuffle=True, drop_last=True,
                                                  num_workers=args.num_worker)
    val_loader = dgl.dataloading.DataLoader(val_g, val_g.nodes(), sampler, device=device,
                                            batch_size=args.batch_size, shuffle=True, drop_last=False,
                                            num_workers=args.num_worker)
    test_loader = dgl.dataloading.DataLoader(test_g, test_g.nodes(), sampler, device=device,
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
