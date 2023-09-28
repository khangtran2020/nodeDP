import dgl
import torch
import torch.nn.functional as F
from functools import partial
from Data.facebook import Facebook
from Data.amazon import Amazon
from copy import deepcopy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from Utils.utils import get_index_by_value, get_index_by_not_list
from rich import print as rprint
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from torch_geometric.transforms import Compose, RandomNodeSplit
from Attacks.Model.train_eval import get_entropy
from Data.dataloader import NodeDataLoader

min_count = {
    'reddit': 10000,
    'facebook': 1000,
    'amazon': 6000
}

def read_data(args, history, hist_exist):
    
    # get graph
    graph = read_graph(data_name=args.dataset)

    if hist_exist == False:

        idx_remain_nodes, idx_sh_nodes, remain_graph, shadow_graph = preprocessing_graph(graph=graph, data_name=args.dataset, 
                                                                                        n_neighbor=args.n_neighbor, n_layers=args.n_layers)
        history['shadow_id'] = idx_sh_nodes.tolist()
        history['remain_id'] = idx_remain_nodes.tolist()
        history['remain_graph'] = {
            'train_id': get_index_by_value(a=remain_graph.ndata['train_mask'], val=1).tolist(),
            'val_id': get_index_by_value(a=remain_graph.ndata['val_mask'], val=1).tolist(),
            'test_id': get_index_by_value(a=remain_graph.ndata['test_mask'], val=1).tolist()
        }

        history['shadow_graph'] = {
            'train_id': get_index_by_value(a=shadow_graph.ndata['sha_train_mask'], val=1).tolist(),
            'val_id': get_index_by_value(a=shadow_graph.ndata['sha_val_mask'], val=1).tolist(),
            'test_id': get_index_by_value(a=shadow_graph.ndata['sha_test_mask'], val=1).tolist()
        }
    
    else:
        remain_graph, shadow_graph = init_graph_from_hist(graph=graph, history=history)

    args.num_feat = remain_graph.ndata['feat'].size(dim=1)
    args.num_class = remain_graph.ndata['label'].max().item()+1

    return remain_graph, shadow_graph

def preprocessing_graph(graph, data_name, n_neighbor, n_layers):

    # filter class by count
    min_cnt = min_count[data_name] if data_name in min_count.keys() else 0
    label_list = filter_class_by_count(graph=graph, min_count=min_cnt)
    num_label = len(label_list)
    idx = torch.index_select(graph.nodes(), 0, graph.ndata['label_mask'].nonzero().squeeze()).numpy()
    graph = graph.subgraph(torch.LongTensor(idx))

    # drop isolated node & self loop
    graph = drop_isolated_node(graph)
    graph = dgl.remove_self_loop(graph)

    # shadow separation
    org_node = graph.nodes()
    num_node = org_node.size(dim=0)
    num_neigh_per_node = (n_neighbor**(n_layers+1) - 1) / (n_neighbor - 1)
    num_node_sh = int(num_node / (2 * num_neigh_per_node))
    num_pt_per_class = int(num_node_sh / num_label)

    rprint(f"The current setting: Org node {num_node}, Shadow node {num_node_sh}, Num node per class {num_pt_per_class}")

    sh_nodes = sampling_shadow_nodes_by_label(graph=graph, num_node_per_class=num_pt_per_class)
    sh_khop_subg = generate_khop_neighbor(graph=graph, nodes=sh_nodes, num_hops=n_layers, num_neigh_per_hop=n_neighbor)

    graph.ndata['shadow_graph'] = torch.zeros(num_node)
    graph.ndata['remain_graph'] = torch.ones(num_node)

    graph.ndata['shadow_graph'][sh_khop_subg] = 1
    graph.ndata['remain_graph'][sh_khop_subg] = 0

    idx_sh_nodes = get_index_by_value(a=graph.ndata['shadow_graph'], val=1)
    idx_remain_nodes = get_index_by_value(a=graph.ndata['remain_graph'], val=1)

    shadow_nodes = org_node[idx_sh_nodes]
    remain_nodes = org_node[idx_remain_nodes]

    shadow_graph = graph.subgraph(shadow_nodes)
    remain_graph = graph.subgraph(remain_nodes)

    rprint(f"Finished shadow separation: shadow graph has {shadow_graph.nodes().size(dim=0)} nodes, and remain graph has {remain_graph.nodes().size(dim=0)}")
    
    # train test split
    node_split(graph=remain_graph, val_size=0.1, test_size=0.15, mode='remain')
    node_split(graph=shadow_graph, val_size=0.1, test_size=0.4, mode='shadow')

    return idx_remain_nodes, idx_sh_nodes, remain_graph, shadow_graph

def read_graph(data_name):

    if data_name == 'reddit':
        data = dgl.data.RedditDataset()
        graph = data[0]
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
    elif data_name == 'cora':
        data = CoraGraphDataset()
        graph = data[0]
        del(graph.ndata['train_mask'])
        del(graph.ndata['val_mask'])
        del(graph.ndata['test_mask'])
    elif data_name == 'citeseer':
        data = CiteseerGraphDataset()
        graph = data[0]
        del(graph.ndata['train_mask'])
        del(graph.ndata['val_mask'])
        del(graph.ndata['test_mask'])
    elif data_name == 'pubmed':
        data = PubmedGraphDataset()
        graph = data[0]
        del(graph.ndata['train_mask'])
        del(graph.ndata['val_mask'])
        del(graph.ndata['test_mask'])

    del(graph.ndata['train_mask'])
    del(graph.ndata['val_mask'])
    del(graph.ndata['test_mask'])

    return graph
    
def filter_class_by_count(graph, min_count):
    target = deepcopy(graph.ndata['label'])
    rprint(f"The target distribution of the original graph is as follows: \n {target.unique(return_counts=True)[1]} \n")
    counts = target.unique(return_counts=True)[1] > min_count
    index = get_index_by_value(a=counts, val=True)
    label_dict = dict(zip(index.tolist(), range(len(index))))
    mask = target.apply_(lambda x: x in index.tolist())
    graph.ndata['label'].apply_(lambda x: label_dict[x] if x in label_dict.keys() else -1)
    graph.ndata['label_mask'] = mask
    return index.tolist()

def drop_isolated_node(graph):
    mask = torch.zeros_like(graph.nodes())
    src, dst = graph.edges()
    mask[src.unique()] = 1
    mask[dst.unique()] = 1
    index = get_index_by_value(a=mask, val=1)
    nodes_id = graph.nodes()[index]
    return graph.subgraph(torch.LongTensor(nodes_id))

def generate_khop_neighbor(graph, nodes, num_hops, num_neigh_per_hop):

    temp = nodes.clone()
    curr = nodes.clone()

    for k in range(num_hops):
        g = dgl.sampling.sample_neighbors(graph, curr, num_neigh_per_hop)
        mask = torch.zeros_like(g.nodes())
        src, dst = g.edges()
        mask[src.unique()] = 1
        mask[dst.unique()] = 1
        index = get_index_by_value(a=mask, val=1)
        new_node = g.nodes()[index]
        idx = get_index_by_not_list(arr=new_node, test_arr=temp)
        curr = new_node[idx]
        temp = new_node

    return temp
        
def sampling_shadow_nodes_by_label(graph, num_node_per_class):

    y = graph.ndata['label']
    num_classes = y.max().item() + 1
    shadow_mask = torch.zeros_like(y)

    for c in range(num_classes):
        idx = (y == c).nonzero(as_tuple=False).view(-1)
        num_nodes = idx.size(0)
        if num_nodes >= num_node_per_class:
            idx = idx[torch.randperm(idx.size(0))][:num_node_per_class]
            shadow_mask[idx[:num_node_per_class]] = 1

    rprint(f"Size of the shadow mask: {shadow_mask.sum()}")
    graph.ndata['shadow_mask'] = shadow_mask
    idx = get_index_by_value(a = shadow_mask, val=1)
    shadow_nodes = graph.nodes()[idx]
    rprint(f"Num node for shadow data {shadow_mask.sum()}")
    return shadow_nodes

def node_split(graph, val_size, test_size, mode='remain'):

    if mode == 'remain':

        num_node = graph.nodes().size(dim=0)
        node_id = range(num_node)
        node_label = graph.ndata['label'].tolist()
        id_train, id_test, y_train, y_test = train_test_split(node_id, node_label, test_size=test_size,
                                                              stratify=node_label)
        
        test_mask = torch.zeros(num_node)
        test_mask[id_test] = 1
        
        id_train, id_val, y_train, y_val = train_test_split(id_train, y_train, test_size=val_size, stratify=y_train)

        train_mask = torch.zeros(num_node)
        train_mask[id_train] = 1

        val_mask = torch.zeros(num_node)
        val_mask[id_val] = 1

        graph.ndata['train_mask'] = train_mask.int()
        graph.ndata['val_mask'] = val_mask.int()
        graph.ndata['test_mask'] = test_mask.int()

        num_tr = graph.ndata['train_mask'].sum()
        num_va = graph.ndata['val_mask'].sum()
        num_te = graph.ndata['test_mask'].sum()

    else:

        node_id = get_index_by_value(a=graph.ndata['shadow_mask'], val=1)
        num_node = node_id.size(dim=0)
        num_te = int(test_size * num_node)
        num_va = int(val_size * num_node)
        num_tr = num_node - num_va - num_te
        perm = torch.randperm(n=num_node)
        node_id = node_id[perm]

        tr_id = node_id[:num_tr]
        va_id = node_id[num_tr:num_tr+num_va]
        te_id = node_id[num_tr+num_va:]

        graph.ndata['sha_train_mask'] = torch.zeros(graph.nodes().size(dim=0))
        graph.ndata['sha_val_mask'] = torch.zeros(graph.nodes().size(dim=0))
        graph.ndata['sha_test_mask'] = torch.zeros(graph.nodes().size(dim=0))

        graph.ndata['sha_train_mask'][tr_id] = 1
        graph.ndata['sha_val_mask'][va_id] = 1
        graph.ndata['sha_test_mask'][te_id] = 1

    rprint(f"Done splitting for {mode} graph: train {num_tr} nodes, val {num_va} nodes, test {num_te} nodes")
       
def init_graph_from_hist(graph, history):

    num_node = graph.nodes().size(dim=0)

    shadow_id = torch.LongTensor(history['shadow_id'])
    remain_id = torch.LongTensor(history['remain_id'])

    shadow_graph = graph.subgraph(shadow_id)
    remain_graph = graph.subgraph(remain_id)

    remain_graph.ndata['train_mask'] = torch.LongTensor(history['remain_graph']['train_id'])
    remain_graph.ndata['val_mask'] = torch.LongTensor(history['remain_graph']['val_id'])
    remain_graph.ndata['test_mask'] = torch.LongTensor(history['remain_graph']['test_id'])

    shadow_graph.ndata['sha_train_mask'] = torch.LongTensor(history['shadow_graph']['train_id'])
    shadow_graph.ndata['sha_val_mask'] = torch.LongTensor(history['shadow_graph']['val_id'])
    shadow_graph.ndata['sha_test_mask'] = torch.LongTensor(history['shadow_graph']['test_id'])

    return remain_graph, shadow_graph

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

def init_shadow_loader(args, device, graph):
    tr_nid = get_index_by_value(a=graph.ndata['sha_train_mask'], val=1).to(device)
    va_nid = get_index_by_value(a=graph.ndata['sha_val_mask'], val=1).to(device)

    sampler = dgl.dataloading.NeighborSampler([args.n_neighbor for i in range(args.n_layers)])

    tr_loader = dgl.dataloading.DataLoader(graph.to(device), tr_nid.to(device), sampler, device=device,
                                           batch_size=args.batch_size, shuffle=True, drop_last=True,
                                           num_workers=args.num_worker)

    va_loader = dgl.dataloading.DataLoader(graph.to(device), va_nid.to(device), sampler, device=device,
                                           batch_size=args.batch_size, shuffle=False, drop_last=False,
                                           num_workers=args.num_worker)
    return tr_loader, va_loader

def generate_attack_samples(graph, conf, mode, device):

    tr_mask = 'train_mask' if mode == 'target' else 'sha_train_mask'
    te_mask = 'test_mask' if mode == 'target' else 'sha_test_mask'

    if mode != 'target':
        num_classes = graph.size(1)
        rprint(f"Existing label: {graph.ndata['label'].unique()}, # class: {num_classes}")
        num_train = graph.ndata[tr_mask].sum()
        num_test = graph.ndata[te_mask].sum()
        num_half = min(num_train, num_test)

        labels = F.one_hot(graph.ndata['label'], num_classes).float().to(device)
        samples = torch.cat([conf, labels], dim=1).to(device)

        perm = torch.randperm(num_train, device=device)[:num_half]
        idx = get_index_by_value(a=graph.ndata[tr_mask], val=1)
        pos_samples = samples[idx][perm]

        perm = torch.randperm(num_test, device=device)[:num_half]
        idx = get_index_by_value(a=graph.ndata[te_mask], val=1)
        neg_samples = samples[idx][perm]

        pos_entropy = get_entropy(pred=pos_samples[:, :num_classes]).mean()
        neg_entropy = get_entropy(pred=neg_samples[:, :num_classes]).mean()
        rprint(f'On graph {mode}, pos_entropy: {pos_entropy:.4f}, neg_entropy: {neg_entropy:.4f}')

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
        num_classes = graph.size(1)
        rprint(f"Existing label: {graph.ndata['label'].unique()}, # class: {num_classes}")

        num_train = graph.ndata[tr_mask].sum()
        num_test = graph.ndata[te_mask].sum()
        num_half = min(num_train, num_test)

        labels = F.one_hot(graph.ndata['label'], num_classes).float().to(device)
        tr_samples = torch.cat([conf, labels], dim=1).to(device)

        labels = F.one_hot(graph.ndata['label'], num_classes).float().to(device)
        te_samples = torch.cat([conf, labels], dim=1).to(device)

        # samples = torch.cat((tr_samples, te_samples), dim=0).to(device)

        perm = torch.randperm(num_train, device=device)[:num_half]
        idx = get_index_by_value(a=graph.ndata[tr_mask], val=1)
        pos_samples = tr_samples[idx][perm]

        perm = torch.randperm(num_test, device=device)[:num_half]
        idx = get_index_by_value(a=graph.ndata[te_mask], val=1)
        neg_samples = te_samples[idx][perm]

        pos_entropy = get_entropy(pred=pos_samples[:, :num_classes]).mean()
        neg_entropy = get_entropy(pred=neg_samples[:, :num_classes]).mean()

        rprint(f'On graph {mode}, pos_entropy: {pos_entropy:.4f}, neg_entropy: {neg_entropy:.4f}')

        x = torch.cat([neg_samples, pos_samples], dim=0)
        y = torch.cat([
            torch.zeros(num_half, dtype=torch.long, device=device),
            torch.ones(num_half, dtype=torch.long, device=device),
        ])

        # shuffle data
        perm = torch.randperm(2 * num_half, device=device)
        x, y = x[perm], y[perm]

        return x, y
    
def init_loader(args, device, train_g, test_g, val_g):
    train_nodes = train_g.nodes()
    val_nodes = val_g.nodes()
    test_nodes = test_g.nodes()

    print('Nodes:', train_g.nodes().size(), val_g.nodes().size(), test_g.nodes().size())
    print('Edges:', train_g.edges()[0].size(), val_g.edges()[0].size(), test_g.edges()[0].size())
    print('Num label:', args.num_class)
    print('Test label dist:', test_g.ndata['label'].unique(return_counts=True))

    sampler = dgl.dataloading.NeighborSampler([args.n_neighbor for i in range(args.n_layers)])
    if args.mode in ['nodedp', 'grad']:
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


