import dgl
import torch

from sklearn.model_selection import train_test_split
from Data.sampler import ComputeSubgraphSampler



def read_data(args, data_name, ratio):
    if data_name == 'cora':
        data = dgl.data.CoraGraphDataset()
    elif data_name == 'citeseer':
        data = dgl.data.CiteseerGraphDataset()
    elif data_name == 'pubmed':
        data = dgl.data.PubmedGraphDataset()
    args.num_feat = data[0].ndata['feat'].shape[1]
    args.num_class = data.num_classes
    g_train, g_test = graph_split(data=data, ratio=ratio)
    return g_train, g_test


def graph_split(data, ratio):
    g = data[0]
    node_id = g.nodes().numpy()
    node_label = g.ndata['label'].numpy()
    id_train, id_test, y_train, y_test = train_test_split(node_id, node_label, test_size=ratio, stratify=node_label)
    train_g = g.subgraph(torch.LongTensor(id_train))
    test_g = g.subgraph(torch.LongTensor(id_test))
    return train_g, test_g


def init_loader(args, device, train_g, val_g, test_g):
    sampler = dgl.dataloading.NeighborSampler([args.n_neighbor for i in range(args.n_layer)])
    if args.mode == 'dp':
        tr_sampler = ComputeSubgraphSampler(num_neighbors=[args.n_neighbor for i in range(args.n_layer)], device=device)
        train_loader = dgl.dataloading.DataLoader(train_g, train_g.nodes(), tr_sampler, device=device,
                                                  batch_size=args.batch_size, shuffle=True, drop_last=True,
                                                  num_workers=args.num_worker)
    else:
        train_loader = dgl.dataloading.DataLoader(train_g, train_g.nodes(), sampler, device=device,
                                                  batch_size=args.batch_size, shuffle=True, drop_last=True,
                                                  num_workers=args.num_worker)
    val_loader = dgl.dataloading.DataLoader(val_g, val_g.nodes(), sampler, device=device,
                                            batch_size=args.batch_size, shuffle=True, drop_last=True,
                                            num_workers=args.num_worker)
    test_loader = dgl.dataloading.DataLoader(test_g, test_g.nodes(), sampler, device=device,
                                             batch_size=args.batch_size, shuffle=True, drop_last=True,
                                             num_workers=args.num_worker)
    return train_loader, val_loader, test_loader
