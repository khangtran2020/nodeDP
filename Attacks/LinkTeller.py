from collections import defaultdict
import numpy as np
from sklearn import metrics
import time
from tqdm import tqdm
import torch
import dgl
from copy import deepcopy

class Attacker:

    def __init__(self, args, graph, model, n_samples, influence, device):
        self.model = model.to(device)
        self.graph = graph.to(device)
        self.graph_ = deepcopy(self.graph)
        self.args = args
        self.graph = dgl.add_self_loop(self.graph)
        self.n_node = self.graph.ndata['feat'].shape[0]
        self.adj = self.graph.adj(scipy_fmt='csr')
        self.features = self.graph.ndata['feat']
        self.n_samples = n_samples
        self.influence = influence
        self.device = device
        # print(self.adj.shape, self.adj.indices, self.adj.indptr)

    # def get_gradient_eps(self, u, v):
    #     pert_1 = torch.zeros_like(self.features)
    #     pert_1[v] = self.features[v] * self.influence
    #     grad = (self.model(self.graph, self.features + pert_1).detach() -
    #             self.model(self.graph, self.features).detach()) / self.influence
    #
    #     return grad[u]

    def get_gradient_eps_mat(self, v):
        pert_1 = torch.zeros_like(self.features)
        pert_1[v] = self.features[v] * self.influence

        self.graph_.ndata['feat'] = deepcopy(self.graph.ndata['feat'] + pert_1)

        sampler = dgl.dataloading.NeighborSampler([self.args.n_neighbor for i in range(self.args.n_layers)])
        loader = dgl.dataloading.DataLoader(self.graph, self.graph.nodes(), sampler, device=self.device,
                                            batch_size=self.args.batch_size, shuffle=False, drop_last=False,
                                            num_workers=self.args.num_worker)

        with torch.no_grad():
            tar_conf = None
            bi = 0
            for d in loader:
                input_nodes, output_nodes, mfgs = d
                inputs = mfgs[0].srcdata["feat"]
                predictions = self.model(mfgs, inputs)
                if bi == 0:
                    tar_conf = predictions
                else:
                    tar_conf = torch.cat((tar_conf, predictions), dim=0)
                bi += 1

        loader_ = dgl.dataloading.DataLoader(self.graph_, self.graph_.nodes(), sampler, device=self.device,
                                            batch_size=self.args.batch_size, shuffle=False, drop_last=False,
                                            num_workers=self.args.num_worker)

        with torch.no_grad():
            tar_conf_ = None
            bi = 0
            for d in loader_:
                input_nodes, output_nodes, mfgs = d
                inputs = mfgs[0].srcdata["feat"]
                predictions = self.model(mfgs, inputs)
                if bi == 0:
                    tar_conf_ = predictions
                else:
                    tar_conf_ = torch.cat((tar_conf_, predictions), dim=0)
                bi += 1
        grad = (tar_conf_ - tar_conf) / self.influence
        return grad

    def link_prediction_attack_efficient(self):
        norm_exist = []
        norm_nonexist = []

        t = time.time()

        # 2. compute influence value for all pairs of nodes
        influence_val = np.zeros((self.n_samples, self.n_samples))

        with torch.no_grad():

            for i in tqdm(range(self.n_samples)):
                u = self.test_nodes[i]
                grad_mat = self.get_gradient_eps_mat(u)

                for j in range(self.n_samples):
                    v = self.test_nodes[j]

                    grad_vec = grad_mat[v]

                    influence_val[i][j] = grad_vec.norm().item()

            print(f'time for predicting edges: {time.time() - t}')

        node2ind = {node: i for i, node in enumerate(self.test_nodes)}

        for u, v in self.exist_edges:
            i = node2ind[u]
            j = node2ind[v]

            norm_exist.append(influence_val[j][i])

        for u, v in self.nonexist_edges:
            i = node2ind[u]
            j = node2ind[v]

            norm_nonexist.append(influence_val[j][i])

        self.compute_and_save(norm_exist, norm_nonexist)

    def compute_and_save(self, norm_exist, norm_nonexist):
        y = [1] * len(norm_exist) + [0] * len(norm_nonexist)
        pred = norm_exist + norm_nonexist
        print('number of prediction:', len(pred))

        fpr, tpr, thresholds = metrics.roc_curve(y, pred)
        print('auc =', metrics.auc(fpr, tpr))

        precision, recall, thresholds_2 = metrics.precision_recall_curve(y, pred)
        print('ap =', metrics.average_precision_score(y, pred))

        folder_name = self.args.save_path
        filename = 'attack_result.pt'
        torch.save({
            'auc': {
                'fpr': fpr,
                'tpr': tpr,
                'thresholds': thresholds
            },
            'pr': {
                'precision': precision,
                'recall': recall,
                'thresholds': thresholds_2
            },
            'result': {
                'y': y,
                'pred': pred,
            }
        }, folder_name + filename)
        print(f'attack results saved to: {filename}')

    def construct_edge_sets_from_random_subgraph(self):
        indices = self.adj.indices
        indptr = self.adj.indptr
        n_nodes = self.adj.shape[0]
        indice_all = range(n_nodes)
        print('#indice =', len(indice_all))
        nodes = np.random.choice(indice_all, self.n_samples, replace=False)  # choose from low degree nodes
        self.test_nodes = nodes
        self.exist_edges, self.nonexist_edges = self._get_edge_sets_among_nodes(indices=indices, indptr=indptr,
                                                                                     nodes=nodes)

    def _get_edge_sets_among_nodes(self, indices, indptr, nodes):
        # construct edge list for each node
        dic = defaultdict(list)

        for u in nodes:
            begg, endd = indptr[u: u + 2]
            dic[u] = indices[begg: endd]

        n_nodes = len(nodes)
        edge_set = []
        nonedge_set = []
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                u, v = nodes[i], nodes[j]
                if v in dic[u]:
                    edge_set.append((u, v))
                else:
                    nonedge_set.append((u, v))

        index = np.arange(len(nonedge_set))
        index = np.random.choice(index, len(edge_set), replace=False)
        print(len(index))
        reduce_nonedge_set = [nonedge_set[i] for i in index]
        print('#nodes =', len(nodes))
        print('#edges_set =', len(edge_set))
        print('#nonedge_set =', len(reduce_nonedge_set))
        return edge_set, reduce_nonedge_set
