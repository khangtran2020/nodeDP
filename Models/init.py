from Models.models import GraphSAGE, GAT, GIN
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
    elif args.model_type == 'gin':
        model = GIN(in_feats=args.num_feat, n_hidden=args.hid_dim, n_classes=args.num_class, n_layers=args.n_layers,
                    aggregator_type=args.aggregator_type, dropout=args.dropout)
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
