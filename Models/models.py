import dgl
import dgl.nn.pytorch as dglnn
import torch.nn
from torch import nn

class GraphSAGE(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.SAGEConv(in_feats, n_hidden, 'mean'))
        for i in range(0, n_layers - 1):
            self.layers.append(dglnn.SAGEConv(n_hidden, n_hidden, 'mean'))
        self.layers.append(nn.Linear(n_hidden, n_classes))
        self.dropout = nn.Dropout(dropout)
        self.n_layers = n_layers
        self.activation = torch.nn.ReLU()

    def forward(self, blocks, x):
        h = x
        for i in range(0, self.n_layers):
            # print(blocks[i].srcdata[dgl.NID].size(), blocks[i].dstdata[dgl.NID].size())
            h = self.layers[i](blocks[i], h)
            h = self.activation(h)
            # h = self.dropout(h)

        h = self.layers[-1](h)
        return h

class GAT(nn.Module):
    def __init__(self, in_feats, n_hidden, n_classes, n_layers, num_head=8, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(dglnn.GATConv(in_feats, n_hidden, num_heads=num_head))
        for i in range(0, n_layers - 1):
            self.layers.append(dglnn.GATConv(n_hidden, n_hidden, num_heads=num_head))
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