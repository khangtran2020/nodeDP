import dgl
import dgl.nn.pytorch as dglnn
import torch.nn
from torch import nn
import dgl.function as fn
import torch.nn.functional as F
from rich import print as rprint

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
                h_dst = h[:blocks[i].num_dst_nodes()]
                h = self.layers[i](blocks[i], (h, h_dst))
                h = self.activation(h)
                out_dict[f'out_{i}'] = h[0]
            h_dst = h[:blocks[-1].num_dst_nodes()]
            h = self.layers[-1](blocks[-1], (h, h_dst))
            pred = self.last_activation(h)
            out_dict[f'out_{self.n_layers-1}'] = pred[0]
            return out_dict, pred
        else:
            h = x
            h_dst = h[:blocks[0].num_dst_nodes()]
            pred = self.last_activation(self.layer(blocks[0], (h, h_dst)))
            out_dict[f'out_{self.n_layers-1}'] = pred[0]
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
                if self.dropout is not None:
                    h = self.dropout(h)
            h = self.layers[-1](h)
            # h = self.last_activation(h)
            return h
        else:
            h = x
            h = self.out_layer(h)
            if self.dropout is not None:
                h = self.dropout(h)
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

class WbAttacker(nn.Module):

    def __init__(self, label_dim, loss_dim, out_dim_list, grad_dim_list, out_keys, model_keys, num_filters, device):
        
        super(WbAttacker, self).__init__()
        self.out_keys = out_keys
        self.model_keys = model_keys
        self.block_label = nn.Sequential(nn.Linear(label_dim, 128),
                                             nn.ReLU(),
                                             nn.Dropout(p=0.2),
                                             nn.Linear(128, 64),
                                             nn.ReLU())
        
        self.block_loss = nn.Sequential(nn.Linear(loss_dim, 128),
                                            nn.ReLU(),
                                            nn.Dropout(p=0.2),
                                            nn.Linear(128, 64),
                                            nn.ReLU())
        
        self.block_out = nn.ModuleDict()
        for i, key in enumerate(out_keys):
            self.block_out[key] = nn.Sequential(nn.Linear(out_dim_list[i], 128),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(128, 64),
                                       nn.ReLU())
        
        self.block_grad = nn.ModuleDict()
        for i, key in enumerate(model_keys):
            grad_dim = grad_dim_list[i]
            self.block_grad[key.replace(".", "-")] = nn.Sequential(nn.Conv2d(1, num_filters, (1, grad_dim[0]), stride=1),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.2),
                                       nn.Flatten(),
                                       nn.Linear(num_filters * grad_dim[1], 128),
                                       nn.ReLU(),
                                       nn.Dropout(p=0.2),
                                       nn.Linear(128, 64),
                                       nn.ReLU())
            
        encoder_input_size = 64 * (len(out_dim_list) + len(grad_dim_list)) + 128
        self.encoder = nn.Sequential(nn.Dropout(p=0.2),
                                     nn.Linear(encoder_input_size, 256),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.2),
                                     nn.Linear(256, 128),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.2),
                                     nn.Linear(128, 64),
                                     nn.ReLU(),
                                     nn.Dropout(p=0.2),
                                     nn.Linear(64, 1))
        
    def forward(self, x):

        label, loss, out_dict, grad_dict = x
         
        label_emb = self.block_label(label)
        loss_emb = self.block_loss(loss)

        overall_emb = torch.cat((label_emb, loss_emb), dim=1)
        for key in self.out_keys:
            out_emb = self.block_out[key](out_dict[key])
            overall_emb = torch.cat((overall_emb, out_emb), dim=1)

        for key in self.model_keys:
            # rprint(f"Forwarding at key {key}, with size {grad_dict[key].size()}")
            grad_emb = self.block_grad[key](grad_dict[key])
            overall_emb = torch.cat((overall_emb, grad_emb), dim=1)

        pred = self.encoder(overall_emb)
        return pred
        

class DotPredictor(nn.Module):
    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            # Compute a new edge feature named 'score' by a dot-product between the
            # source node feature 'h' and destination node feature 'h'.
            g.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            # u_dot_v returns a 1-element vector for each edge so you need to squeeze it.
            return g.edata['score'][:, 0]