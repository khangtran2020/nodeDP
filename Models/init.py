from Models.models import GraphSAGE, GAT, NN
from torch.optim import Adam, AdamW, SGD


def init_model(args):
    print("Training with graph {}".format(args.model_type))
    model = None
    if args.model_type == 'sage':
        model = GraphSAGE(in_feats=args.num_feat, n_hidden=args.hid_dim, n_classes=args.num_class,
                          n_layers=args.n_layers, dropout=args.dropout, aggregator_type=args.aggregator_type)
    elif args.model_type == 'gat':
        model = GAT(in_feats=args.num_feat, n_hidden=args.hid_dim, n_classes=args.num_class, n_layers=args.n_layers,
                    num_head=args.num_head, dropout=args.dropout)
    elif args.model_type == 'mlp':
        model = NN(input_dim=args.num_feat, n_layer=args.n_layers, hidden_dim=args.hid_dim,output_dim=args.num_class)
    return model


def init_optimizer(optimizer_name, model, lr):
    print("Optimizing with optimizer {}".format(optimizer_name))
    if optimizer_name == 'adam':
        optimizer = Adam(model.parameters(), lr=lr)
    elif optimizer_name == 'adamw':
        optimizer = AdamW(model.parameters(), lr=lr)
    elif optimizer_name == 'sgd':
        optimizer = SGD(model.parameters(), lr=lr)
    return optimizer
