import dgl
import dgl.nn.pytorch as dglnn
import torch.nn
from torch import nn
import dgl.function as fn
import torch.nn.functional as F
from rich import print as rprint
from dgl.dataloading.base import NID

class GraphSAGE(nn.Module):

    def __init__(self, in_feats, n_hidden, n_classes, n_layers, dropout=0.2, aggregator_type='mean'):
        
        super().__init__()

        self.n_layers = n_layers
        if n_layers > 1:
            self.layers = nn.ModuleList()
            self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, aggregator_type))
            for i in range(0, n_layers - 2):
                self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, aggregator_type))
            self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, aggregator_type))
            self.dropout = nn.Dropout(dropout)
            self.activation = torch.nn.SELU()
            self.last_activation = torch.nn.Softmax(dim=1) if n_classes > 1 else torch.nn.Sigmoid()
            print(f"Using activation for last layer {self.last_activation}")
        else:
            self.layer = dglnn.SAGEConv(in_feats, n_classes, aggregator_type)
            self.dropout = nn.Dropout(dropout)
            self.activation = torch.nn.SELU()
            self.last_activation = torch.nn.Softmax(dim=1) if n_classes > 1 else torch.nn.Sigmoid()
            print(f"Using activation for last layer {self.last_activation}")

    def forwardout(self, blocks, x):
        out_dict = {}
        if self.n_layers > 1:
            h = x
            for i in range(0, self.n_layers-1):
                rprint(f"At layer {i}, block {i} has {blocks[i].srcdata[NID]} src nodes and {blocks[i].dstdata[NID]} dst nodes")
                rprint(f"With feature size {blocks[i].ndata['feat']['_N'].size()}\n")
                h_dst = h[:blocks[i].num_dst_nodes()]
                h = self.layers[i](blocks[i], (h, h_dst))
                h = self.activation(h)
                out_dict[f'out_{i}'] = h
            h_dst = h[:blocks[-1].num_dst_nodes()]
            h = self.layers[-1](blocks[-1], (h, h_dst))
            rprint(f"At layer {self.n_layers-1}, block {self.n_layers-1} has {blocks[-1].srcdata[NID]} src nodes and {blocks[-1].dstdata[NID]} dst nodes")
            rprint(f"With feature size {blocks[-1].ndata['feat']['_N'].size()}\n")
            pred = self.last_activation(h)
            out_dict[f'out_{self.n_layers-1}'] = pred
            return out_dict, pred
        else:
            h = x
            h_dst = h[:blocks[0].num_dst_nodes()]
            pred = self.last_activation(self.layer(blocks[0], (h, h_dst)))
            out_dict[f'out_{self.n_layers-1}'] = pred
            return out_dict, pred

    def forward(self, blocks, x):
        if self.n_layers > 1:
            h = x
            for i in range(0, self.n_layers-1):
                h_dst = h[:blocks[i].num_dst_nodes()]
                h = self.layers[i](blocks[i], (h, h_dst))
                h = self.activation(h)
            h_dst = h[:blocks[-1].num_dst_nodes()]
            h = self.last_activation(self.layers[-1](blocks[-1], (h, h_dst)))
            return h
        else:
            h = x
            h_dst = h[:blocks[0].num_dst_nodes()]
            h = self.last_activation(self.layer(blocks[0], (h, h_dst)))
            return h
        
    def full(self, g, x):
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
            self.activation = torch.nn.SELU()
            self.last_activation = torch.nn.Softmax(dim=1) if n_classes > 1 else torch.nn.Sigmoid()
            print(f"Using activation for last layer {self.last_activation}")
        else:
            self.layer = dglnn.GATConv(in_feats, n_classes, num_heads=1, allow_zero_in_degree=True)
            self.dropout = nn.Dropout(dropout)
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

    def full(self, g, x):
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

class NN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer, dropout=None):
        super(NN, self).__init__()
        self.n_layers = n_layer
        if self.n_layers > 1:
            self.n_hid = n_layer - 2
            self.layers = nn.ModuleList()
            self.layers.append(nn.Linear(in_features=input_dim, out_features=hidden_dim))
            for i in range(self.n_hid):
                layer = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
                nn.init.kaiming_uniform_(layer.weight, nonlinearity="relu")
                self.layers.append(layer)
            self.layers.append(nn.Linear(in_features=hidden_dim, out_features=output_dim))
        else:
            self.out_layer = nn.Linear(in_features=input_dim, out_features=output_dim)
        
        self.activation = torch.nn.SELU()
        self.last_activation = torch.nn.Softmax(dim=1) if output_dim > 1 else torch.nn.Sigmoid()
        self.dropout = nn.Dropout(dropout) if dropout is not None else None


    def forward(self, x):
        if self.n_layers > 1:
            h = x
            for i in range(0, self.n_layers-1):
                h = self.layers[i](h)
                h = self.activation(h)
            h = self.layers[-1](h)
            # h = self.last_activation(h)
            return h
        else:
            h = x
            h = self.out_layer(h)
            # h = self.last_activation(h)
            return h

class CustomNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layer, dropout=None):
        super(CustomNN, self).__init__()
        self.input_dim = input_dim
        self.block_1 = NN(input_dim=input_dim, hidden_dim= hidden_dim, output_dim=hidden_dim, n_layer=n_layer-1, dropout=dropout)
        self.block_2 = NN(input_dim=input_dim, hidden_dim= hidden_dim, output_dim=hidden_dim, n_layer=n_layer-1, dropout=dropout)
        self.final_block = NN(input_dim=int(2*hidden_dim), hidden_dim= hidden_dim, output_dim=output_dim, n_layer=n_layer-1, dropout=dropout)
        self.last_activation = torch.nn.Softmax(dim=1) if output_dim > 1 else torch.nn.Sigmoid()

    def forward(self, x):
        h1 = x[:,:self.input_dim]
        h2 = x[:,self.input_dim:]
        h1 = self.block_1(h1)
        h2 = self.block_2(h2)
        h = torch.cat((h1, h2), dim=1)
        h = self.final_block(h)
        return self.last_activation(h)

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]