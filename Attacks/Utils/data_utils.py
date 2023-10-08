import dgl
import torch
import numpy as np
from torch.utils.data import Dataset
from torch_geometric.transforms import Compose, RandomNodeSplit
from Utils.utils import get_index_by_value
from sklearn.linear_model import LogisticRegression
from rich import print as rprint
from functools import partial
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
    g = dgl.graph(([], []), num_nodes=len(nodes)).to(device)
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
    graph.ndata['train_mask'] = train_mask
    graph.ndata['train_mask'] = train_mask

    graph.ndata['id_intr'] = (torch.zeros(graph.nodes().size(dim=0)) - 1).int()
    graph.ndata['id_intr'][id_intr] = g_train.nodes().clone().int()

    graph.ndata['id_inte'] = (torch.zeros(graph.nodes().size(dim=0)) - 1).int()
    graph.ndata['id_inte'][id_inte] = g_test.nodes().clone().int()

    idx = torch.index_select(graph.nodes(), 0, graph.ndata['label_mask'].nonzero().squeeze()).numpy()
    graph = graph.subgraph(torch.LongTensor(idx))
    if (args.submode == 'density') and (args.density != 1.0):
        graph = drop_isolated_node(graph)
    args.num_data_point = len(g_train.nodes())
    return g_train, g_val, g_test, graph

def shadow_split_wbextreme(graph, ratio, history=None, exist=False, diag=False):

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

        org_num_node = org_nodes.size(dim=0)
        train_mask = torch.zeros(org_num_node)
        test_mask = torch.zeros(org_num_node)

        pos_mask_tr = torch.zeros(org_num_node)
        pos_mask_te = torch.zeros(org_num_node)

        neg_mask_tr = torch.zeros(org_num_node)
        neg_mask_te = torch.zeros(org_num_node)
        
        pos_mask = torch.zeros(org_num_node)
        neg_mask = torch.zeros(org_num_node)

        membership_label = torch.zeros(org_num_node)

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

        shadow_pos_graph = graph.subgraph(shatr_nodes)
        shadow_neg_graph = graph.subgraph(te_node)
        shadow_nodes = torch.cat((shatr_nodes, te_node), dim=0)
        shadow_graph = dgl.merge([shadow_pos_graph, shadow_neg_graph])

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

        shatr_nodes_idx = get_index_by_value(a=graph.ndata['pos_mask'], val=1)
        te_node_idx = get_index_by_value(a=graph.ndata['neg_mask'], val=1)

        shatr_nodes = graph.nodes()[shatr_nodes_idx]
        te_node = graph.nodes()[te_node_idx]

        shadow_pos_graph = graph.subgraph(shatr_nodes)
        shadow_neg_graph = graph.subgraph(te_node)
        shadow_graph = dgl.merge([shadow_pos_graph, shadow_neg_graph])

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
