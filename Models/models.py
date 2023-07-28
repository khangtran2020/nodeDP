import dgl
import dgl.nn.pytorch as dglnn
import torch.nn
from torch import nn
import dgl.function as fn
import torch.nn.functional as F

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, dropout=0.2, aggregator_type='gcn'):
        super().__init__()
        self.n_layers = n_layers
        if n_layers > 1:
            self.layers = nn.ModuleList()
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(0, n_layers - 2):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
            self.dropout = nn.Dropout(dropout)
            self.batch_norm = torch.nn.BatchNorm1d(n_hidden)
            self.activation = torch.nn.SELU()
            self.last_activation = torch.nn.Softmax(dim=1) if n_classes > 1 else torch.nn.Sigmoid()
            print(f"Using activation for last layer {self.last_activation}")
        else:
            self.layer = dglnn.SAGEConv(in_feats, n_classes, 'mean')
            self.dropout = nn.Dropout(dropout)
            self.batch_norm = torch.nn.BatchNorm1d(n_hidden)
            self.activation = torch.nn.SELU()
            self.last_activation = torch.nn.Softmax(dim=1) if n_classes > 1 else torch.nn.Sigmoid()
            print(f"Using activation for last layer {self.last_activation}")

    def forward(self, blocks, x):
        if self.n_layers > 1:
            h = x
            for i in range(0, self.n_layers-1):
                h_dst = h[:blocks[i].num_dst_nodes()]
                h = self.layers[i](blocks[i], (h, h_dst))
                # h = self.batch_norm(h)
                # h = self.dropout(h)
                h = self.activation(h)
            h_dst = h[:blocks[-1].num_dst_nodes()]
            h = self.last_activation(self.layers[-1](blocks[-1], (h, h_dst)))
            return h
        else:
            h = x
            h_dst = h[:blocks[0].num_dst_nodes()]
            h = self.last_activation(self.layer(blocks[0], (h, h_dst)))
            return h


class GAT(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, num_head=8, dropout=0.2):
        super().__init__()
        self.n_layers = n_layers
        if n_layers > 1:
            self.layers = nn.ModuleList()
            self.layers.append(dglnn.GATConv(in_feats, n_hidden, num_heads=num_head, allow_zero_in_degree=True))
            for i in range(0, n_layers - 1):
                self.layers.append(dglnn.GATConv(n_hidden, n_hidden, num_heads=num_head, allow_zero_in_degree=True))
            self.classification_layer = torch.nn.Linear(in_features=n_hidden, out_features=n_classes)
            self.dropout = nn.Dropout(dropout)
            self.batch_norm = torch.nn.BatchNorm1d(n_hidden)
            self.activation = torch.nn.SELU()
            self.last_activation = torch.nn.Softmax(dim=1) if n_classes > 1 else torch.nn.Sigmoid()
            print(f"Using activation for last layer {self.last_activation}")
        else:
            self.layer = dglnn.GATConv(in_feats, n_classes, num_heads=1, allow_zero_in_degree=True)
            self.dropout = nn.Dropout(dropout)
            self.batch_norm = torch.nn.BatchNorm1d(n_hidden)
            self.activation = torch.nn.SELU()
            self.last_activation = torch.nn.Softmax(dim=1) if n_classes > 1 else torch.nn.Sigmoid()
            print(f"Using activation for last layer {self.last_activation}")

    def forward(self, blocks, x):
        if self.n_layers > 1:
            h = x
            for i in range(0, self.n_layers):
                h_dst = h[:blocks[i].num_dst_nodes()]
                h = self.layers[i](blocks[i], (h, h_dst))
                h = self.activation(h)
            h = h.mean(dim=tuple([i for i in range(1, self.n_layers+1)]))
            h = self.last_activation(self.classification_layer(h))
            return h
        else:
            h = x
            h_dst = h[:blocks[0].num_dst_nodes()]
            h = self.activation(self.layer(blocks[0], (h, h_dst)))
            h = h.mean(dim=(1, 2))
            h = self.last_activation(self.classification_layer(h))
            return h


class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer, dropout=None):
        super(NN, self).__init__()
        self.n_hid = n_layer - 2
        self.in_layer = nn.Linear(in_features=input_dim, out_features=hidden_dim)
        nn.init.kaiming_uniform_(self.in_layer.weight, nonlinearity="relu")
        self.hid_layer = nn.ModuleList()
        for i in range(self.n_hid):
            layer = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
            nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
            self.hid_layer.append(layer)
        self.out_layer = nn.Linear(in_features=hidden_dim, out_features=output_dim)
        self.dropout = nn.Dropout(dropout) if dropout is not None else None


    def forward(self, x):
        h = torch.nn.functional.relu(self.in_layer(x))
        for i in range(self.n_hid):
            h = self.dropout(h) if self.dropout is not None else h
            h = torch.nn.functional.relu(self.hid_layer[i](h))

        h = torch.nn.functional.sigmoid(self.out_layer(h))
        return h

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]


class GATFull(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, num_head=8, dropout=0.2):
        super().__init__()
        self.n_layers = n_layers
        if n_layers > 1:
            self.layers = nn.ModuleList()
            self.layers.append(dglnn.GATConv(in_feats, n_hidden, num_heads=num_head, allow_zero_in_degree=True))
            for i in range(0, n_layers - 1):
                self.layers.append(dglnn.GATConv(n_hidden, n_hidden, num_heads=num_head, allow_zero_in_degree=True))
            self.classification_layer = torch.nn.Linear(in_features=n_hidden, out_features=n_classes)
            self.dropout = nn.Dropout(dropout)
            self.batch_norm = torch.nn.BatchNorm1d(n_hidden)
            self.activation = torch.nn.SELU()
            self.last_activation = torch.nn.Softmax(dim=1) if n_classes > 1 else torch.nn.Sigmoid()
            print(f"Using activation for last layer {self.last_activation}")
        else:
            self.layer = dglnn.GATConv(in_feats, n_classes, num_heads=1, allow_zero_in_degree=True)
            self.dropout = nn.Dropout(dropout)
            self.batch_norm = torch.nn.BatchNorm1d(n_hidden)
            self.activation = torch.nn.SELU()
            self.last_activation = torch.nn.Softmax(dim=1) if n_classes > 1 else torch.nn.Sigmoid()
            print(f"Using activation for last layer {self.last_activation}")

    def forward(self, g, x):
        if self.n_layers > 1:
            h = x
            for i in range(0, self.n_layers):
                h = self.layers[i](g, h)
                h = self.activation(h)
            h = h.mean(dim=tuple([i for i in range(1, self.n_layers+1)]))
            h = self.last_activation(self.classification_layer(h))
            return h
        else:
            h = x
            h = self.activation(self.layer(g, h))
            h = h.mean(dim=(1, 2))
            h = self.last_activation(self.classification_layer(h))
            return h


class GraphSageFull(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, dropout=0.2, aggregator_type='gcn'):
        super().__init__()
        self.n_layers = n_layers
        if n_layers > 1:
            self.layers = nn.ModuleList()
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
            for i in range(0, n_layers - 2):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
            self.dropout = nn.Dropout(dropout)
            self.batch_norm = torch.nn.BatchNorm1d(n_hidden)
            self.activation = torch.nn.SELU()
            self.last_activation = torch.nn.Softmax(dim=1) if n_classes > 1 else torch.nn.Sigmoid()
            print(f"Using activation for last layer {self.last_activation}")
        else:
            self.layer = dglnn.SAGEConv(in_feats, n_classes, 'mean')
            self.dropout = nn.Dropout(dropout)
            self.batch_norm = torch.nn.BatchNorm1d(n_hidden)
            self.activation = torch.nn.SELU()
            self.last_activation = torch.nn.Softmax(dim=1) if n_classes > 1 else torch.nn.Sigmoid()
            print(f"Using activation for last layer {self.last_activation}")

    def forward(self, g, x):
        if self.n_layers > 1:
            h = x
            for i in range(0, self.n_layers-1):
                h = self.layers[i](g, h)
                h = self.activation(h)
            h = self.last_activation(self.layers[-1](g, h))
            return h
        else:
            h = x
            h = self.last_activation(self.layer(g, (g, h)))
            return h