import dgl
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from Utils.utils import get_index_by_value
from sklearn.linear_model import LogisticRegression
from rich import print as rprint


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

def shadow_split(graph, ratio, train_ratio=0.4, test_ratio=0.4, history=None, exist=False):

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
        top_k_conf, _ = torch.topk(conf, k=2)
        top_k_nohop, _ = torch.topk(nohop_conf, k=2)
        samples = torch.cat([top_k_conf, top_k_nohop], dim=1).to(device)

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
        top_k_conf, _ = torch.topk(conf, k=2)
        top_k_nohop, _ = torch.topk(nohop_conf, k=2)
        tr_samples = torch.cat([top_k_conf, top_k_nohop], dim=1).to(device)

        top_k_conf, _ = torch.topk(te_conf, k=2)
        top_k_nohop, _ = torch.topk(te_nohop_conf, k=2)
        te_samples = torch.cat([top_k_conf, top_k_nohop], dim=1).to(device)

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