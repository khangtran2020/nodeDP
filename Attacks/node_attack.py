import torch
import torchmetrics
from Data.read import *
from Models.init import init_model, init_optimizer
from Runs.run_clean import run as run_clean
from Runs.run_nodedp import run as run_nodedp
from Utils.utils import *
from loguru import logger
from rich import print as rprint
from Attacks.train_eval import train_shadow, train_attack, eval_attack_step
from Attacks.helper import generate_attack_samples, Data
from Models.models import NN, GraphSageFull, GATFull

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")


def retrain(args, train_g, val_g, test_g, current_time, device, history):
    train_g = train_g.to(device)
    val_g = val_g.to(device)
    test_g = test_g.to(device)
    rprint(f"Node feature device: {train_g.ndata['feat'].device}, Node label device: {train_g.ndata['feat'].device}, "
           f"Src edges device: {train_g.edges()[0].device}, Dst edges device: {train_g.edges()[1].device}")
    tr_loader, va_loader, te_loader = init_loader(args=args, device=device, train_g=train_g, test_g=test_g,
                                                  val_g=val_g)

    model = init_model(args=args)
    optimizer = init_optimizer(optimizer_name=args.optimizer, model=model, lr=args.lr)
    tar_name = get_name(args=args, current_date=current_time)
    history['name'] = tar_name
    tr_info = (train_g, tr_loader)
    va_info = va_loader
    te_info = (test_g, te_loader)

    if args.tar_clean == 1:
        run_mode = run_clean
    else:
        run_mode = run_nodedp

    tar_model, tar_history = run_mode(args=args, tr_info=tr_info, va_info=va_info, te_info=te_info, model=model,
                                      optimizer=optimizer, name=tar_name, device=device, history=history)
    return tar_model, tar_history


def run_NMI(args, current_time, device):
    with timeit(logger=logger, task='init-target-model'):
        if args.retrain_tar:
            history = init_history_attack()
            train_g, val_g, test_g, graph = read_data(args=args, data_name=args.dataset, history=history)
            tar_model, tar_history = retrain(args=args, train_g=train_g, val_g=val_g, test_g=test_g,
                                             current_time=current_time, history=history, device=device)
        else:
            tar_history = read_pickel(args.res_path + f'{args.tar_name}.pkl')
            train_g, val_g, test_g, graph = read_data_attack(args=args, data_name=args.dataset, history=tar_history)
            tar_model = init_model(args=args)
            tar_model.load_state_dict(torch.load(args.save_path + f'{args.tar_name}.pt'))

    with torch.no_grad():
        if args.model_type == 'sage':
            model = GraphSageFull(in_feats=args.num_feat, n_hidden=args.hid_dim, n_classes=args.num_class,
                                  n_layers=args.n_layers, dropout=args.dropout, aggregator_type=args.aggregator_type)
            model.load_state_dict(torch.load(args.save_path + f"{tar_history['name']}.pt"))
        else:
            model = GATFull(in_feats=args.num_feat, n_hidden=args.hid_dim, n_classes=args.num_class,
                            n_layers=args.n_layers, num_head=args.num_head, dropout=args.dropout)
            model.load_state_dict(torch.load(args.save_path + f"{tar_history['name']}.pt"))

        # device = torch.device('cpu')
        with timeit(logger=logger, task='preparing-shadow-data'):
            # split shadow data
            train_g = train_g.to(device)
            test_g = test_g.to(device)
            model.to(device)

            tr_conf = model(train_g, train_g.ndata['feat'])
            train_g.ndata['tar_conf'] = tr_conf

            te_conf = model(test_g, test_g.ndata['feat'])
            test_g.ndata['tar_conf'] = te_conf

            randomsplit(graph=train_g, num_node_per_class=1000, train_ratio=0.4, test_ratio=0.4)

    with timeit(logger=logger, task='training-shadow-model'):
        # init shadow model
        shadow_model = init_model(args=args)
        shadow_optimizer = init_optimizer(optimizer_name=args.optimizer, model=shadow_model, lr=args.lr)

        # init loader
        tr_loader, va_loader, te_loader = init_shadow_loader(args=args, device=device, graph=train_g)

        # train shadow model
        shadow_model = train_shadow(args=args, tr_loader=tr_loader, va_loader=va_loader, shadow_model=shadow_model,
                                    epochs=args.shaddow_epochs, optimizer=shadow_optimizer, name=tar_history['name'],
                                    device=device)
    with torch.no_grad():
        if args.model_type == 'sage':
            model_shadow = GraphSageFull(in_feats=args.num_feat, n_hidden=args.hid_dim, n_classes=args.num_class,
                                         n_layers=args.n_layers, dropout=args.dropout,
                                         aggregator_type=args.aggregator_type)
            model_shadow.load_state_dict(torch.load(args.save_path + f"{tar_history['name']}_shadow.pt"))
        else:
            model_shadow = GATFull(in_feats=args.num_feat, n_hidden=args.hid_dim, n_classes=args.num_class,
                                   n_layers=args.n_layers, num_head=args.num_head, dropout=args.dropout)
            model_shadow.load_state_dict(torch.load(args.save_path + f"{tar_history['name']}_shadow.pt"))

        with timeit(logger=logger, task='preparing-attack-data'):
            model_shadow.to(device)
            shadow_conf = model_shadow(train_g, train_g.ndata['feat'])
            train_g.ndata['shadow_conf'] = shadow_conf

            x, y = generate_attack_samples(tr_graph=train_g, tr_conf=shadow_conf, mode='shadow', device=device)
            x_test, y_test = generate_attack_samples(tr_graph=train_g, tr_conf=tr_conf, mode='target', device=device,
                                                     te_graph=test_g, te_conf=te_conf)
            x = torch.cat([x, x_test], dim=0)
            y = torch.cat([y, y_test], dim=0)
            num_test = x_test.size(0)
            num_train = int((x.size(0) - num_test) * 0.8)

            new_dim = x.size(dim=1)
            # train test split

            tr_data = Data(X=x[:num_train], y=y[:num_train])
            va_data = Data(X=x[num_train:-num_test], y=y[num_train:-num_test])
            te_data = Data(X=x[-num_test:], y=y[-num_test:])

    device = torch.device('cpu')
    with timeit(logger=logger, task='train-attack-model'):

        tr_loader = torch.utils.data.DataLoader(tr_data, batch_size=args.batch_size,
                                                pin_memory=False, drop_last=True, shuffle=True)

        va_loader = torch.utils.data.DataLoader(va_data, batch_size=args.batch_size, num_workers=0, shuffle=False,
                                                pin_memory=False, drop_last=False)

        te_loader = torch.utils.data.DataLoader(te_data, batch_size=args.batch_size, num_workers=0, shuffle=False,
                                                pin_memory=False, drop_last=False)
        attack_model = NN(input_dim=new_dim, hidden_dim=64, output_dim=1, n_layer=3)
        attack_optimizer = init_optimizer(optimizer_name=args.optimizer, model=attack_model, lr=args.lr)

        attack_model = train_attack(args=args, tr_loader=tr_loader, va_loader=va_loader, te_loader=te_loader,
                                    attack_model=attack_model, epochs=args.attack_epochs, optimizer=attack_optimizer,
                                    name=tar_history['name'], device=device)

    attack_model.load_state_dict(torch.load(args.save_path + f"{tar_history['name']}_attack.pt"))
    te_loss, te_auc = eval_attack_step(model=attack_model, device=device, loader=te_loader,
                                       metrics=torchmetrics.classification.BinaryAUROC().to(device),
                                       criterion=torch.nn.BCELoss())
    rprint(f"Attack AUC: {te_auc}")
