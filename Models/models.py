import dgl
import dgl.nn.pytorch as dglnn
import torch.nn
from torch import nn

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, dropout=0.2, aggregator_type='gcn'):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(0, n_layers - 2):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(dglnn.SAGEConv(n_hidden, n_classes, 'mean'))
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers
        self.batch_norm = torch.nn.BatchNorm1d(n_hidden)
        self.activation = torch.nn.ReLU()
        self.last_activation = torch.nn.Softmax(dim=1) if n_classes > 1 else torch.nn.Sigmoid()
        print(f"Using activation for last layer {self.last_activation}")

    def forward(self, blocks, x):
        h = x
        for i in range(0, self.n_layers-1):
            h_dst = h[:blocks[i].num_dst_nodes()]
            h = self.layers[i](blocks[i], (h, h_dst))
            # h = self.batch_norm(h)
            h = self.dropout(h)
            h = self.activation(h)
            norm = h.norm(p=2, dim=1, keepdim = True)
            h = h.div(norm + 1e-12)
        h_dst = h[:blocks[-1].num_dst_nodes()]
        h = self.last_activation(self.layers[-1](blocks[-1], (h, h_dst)))
        return h

class GAT(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, num_head=8, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GATConv(in_feats, n_hidden, num_heads=num_head))
        for i in range(0, n_layers - 2):
            self.layers.append(dglnn.GATConv(n_hidden, n_hidden, num_heads=num_head))
        self.layers.append(dglnn.GATConv(n_hidden, n_classes, num_heads=num_head))
        self.dropout = nn.Dropout(dropout)
        self.last_activation = torch.nn.Softmax(dim=1) if n_classes > 1 else torch.nn.Sigmoid()
        print(f"Using activation for last layer {self.last_activation}")

    def forward(self, blocks, x):
        h = x
        for i in range(0, self.n_layers - 1):
            # print(blocks[i].srcdata[dgl.NID].size(), blocks[i].dstdata[dgl.NID].size())
            h = self.layers[i](blocks[i], h)
            h = self.activation(h)
            h = self.dropout(h)

        h = self.last_activation(self.layers[-1](blocks[-1], h))
        return h

class GIN(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, aggregator_type='sum', dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GINConv(nn.Linear(in_feats, n_hidden), aggregator_type=aggregator_type))
        for i in range(0, n_layers - 1):
            self.layers.append(dglnn.GINConv(nn.Linear(n_hidden, n_hidden), aggregator_type=aggregator_type))
        self.layers.append(nn.Linear(n_hidden, n_classes))
        self.dropout = nn.Dropout(dropout)

    def forward(self, blocks, x):
        h = x
        for i in range(0, self.n_layers):
            h = self.layers[i](blocks[i], h)
            h = self.activation(h)
            h = self.dropout(h)

        h = self.layers[-1](h)
        return h