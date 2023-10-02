import sys
import torch
import torchmetrics
from loguru import logger
from hashlib import md5
from rich import print as rprint
from Utils.utils import timeit
from Models.models import NN, CustomNN
from Models.init import init_model, init_optimizer
from Attacks.Utils.utils import save_dict
from Attacks.Utils.data_utils import init_shadow_loader, generate_attack_samples, generate_nohop_graph, Data, test_distribution_shift
from Attacks.Utils.train_eval import train_shadow, train_attack, eval_attack_step, retrain

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")


def run(args, graph, model, device, history, name):

    train_g, val_g, test_g = graph
    model_hist, att_hist = history

    with timeit(logger=logger, task='init-target-model'):

        if args.exist_model == False:
            model_name = f"{md5(name['model'].encode()).hexdigest()}.pt"
            model, model_hist = retrain(args=args, train_g=train_g, val_g=val_g, test_g=test_g, model=model, 
                                        device=device, history=model_hist, name=model_name)
            
            target_model_name = f"{md5(name['model'].encode()).hexdigest()}.pkl"
            target_model_path = args.res_path + target_model_name
            save_dict(path=target_model_path, dct=model_hist)

    with timeit(logger=logger, task='preparing-shadow-data'):

        with torch.no_grad():
            train_g = train_g.to(device)
            test_g = test_g.to(device)

            train_g_nohop = generate_nohop_graph(graph=train_g, device=device)
            test_g_nohop = generate_nohop_graph(graph=test_g, device=device)

            model.to(device)

            tr_conf = model.full(train_g, train_g.ndata['feat'])
            tr_conf_nohop = model.full(train_g_nohop, train_g_nohop.ndata['feat'])

            train_g.ndata['tar_conf'] = tr_conf
            train_g.ndata['tar_conf_nohop'] = tr_conf_nohop

            te_conf = model.full(test_g, test_g.ndata['feat'])
            te_conf_nohop = model.full(test_g_nohop, test_g_nohop.ndata['feat'])

            test_g.ndata['tar_conf'] = te_conf
            test_g.ndata['tar_conf_nohop'] = te_conf_nohop

    with timeit(logger=logger, task='training-shadow-model'):

        # init shadow model
        shadow_model = init_model(args=args)
        shadow_model_nohop = init_model(args=args)
        shadow_optimizer = init_optimizer(optimizer_name=args.optimizer, model=shadow_model, lr=args.sha_lr)
        shadow_nohop_optimizer = init_optimizer(optimizer_name=args.optimizer, model=shadow_model_nohop, lr=args.sha_lr)

        # init loader
        tr_loader, te_loader = init_shadow_loader(args=args, device=device, graph=train_g)

        # train shadow model
        shadow_model = train_shadow(args=args, tr_loader=tr_loader, shadow_model=shadow_model,
                                    epochs=args.sha_epochs, optimizer=shadow_optimizer, name=name['att'],
                                    device=device, history=att_hist, mode='hops')
        
        shadow_model_nohop = train_shadow(args=args, tr_loader=tr_loader, shadow_model=shadow_model_nohop,
                                    epochs=args.sha_epochs, optimizer=shadow_nohop_optimizer, name=name['att'],
                                    device=device, history=att_hist, mode='nohop')
        
    with timeit(logger=logger, task='preparing-attack-data'):
    
        shadow_model.load_state_dict(torch.load(args.save_path + f"{name['att']}_hops_shadow.pt"))
        shadow_model_nohop.load_state_dict(torch.load(args.save_path + f"{name['att']}_nohop_shadow.pt"))
        
        with torch.no_grad():

            shadow_model.to(device)
            shadow_model_nohop.to(device)
            shadow_conf = shadow_model.full(train_g, train_g.ndata['feat'])
            shadow_conf_nohop = shadow_model_nohop.full(train_g_nohop, train_g_nohop.ndata['feat'])
            train_g.ndata['shadow_conf'] = shadow_conf

            x, y = generate_attack_samples(graph=train_g, conf=shadow_conf, nohop_conf=shadow_conf_nohop, mode='shadow', device=device)
            x_test, y_test = generate_attack_samples(graph=train_g, conf=tr_conf, nohop_conf=tr_conf_nohop, mode='target', 
                                                     te_graph=test_g, te_conf=te_conf, te_nohop_conf=te_conf_nohop, device=device)
            
            test_distribution_shift(x_tr=x, x_te=x_test)
            x = torch.cat([x, x_test], dim=0)
            y = torch.cat([y, y_test], dim=0)
            for i in range(x.size(dim=1)):
                x[:, i] = (x[:,i] - x[:,i].mean()) / (x[:,i].std() + 1e-12)
            num_test = x_test.size(0)
            num_train = int((x.size(0) - num_test) * 0.8)

            new_dim = int(x.size(dim=1)/2)
            # train test split

            tr_data = Data(X=x[:num_train], y=y[:num_train])
            va_data = Data(X=x[num_train:-num_test], y=y[num_train:-num_test])
            te_data = Data(X=x[-num_test:], y=y[-num_test:])

    # device = torch.device('cpu')
    with timeit(logger=logger, task='train-attack-model'):

        tr_loader = torch.utils.data.DataLoader(tr_data, batch_size=args.att_batch_size,
                                                pin_memory=False, drop_last=True, shuffle=True)

        va_loader = torch.utils.data.DataLoader(va_data, batch_size=args.att_batch_size, num_workers=0, shuffle=False,
                                                pin_memory=False, drop_last=False)

        te_loader = torch.utils.data.DataLoader(te_data, batch_size=args.att_batch_size, num_workers=0, shuffle=False,
                                                pin_memory=False, drop_last=False)
        
        attack_model = CustomNN(input_dim=new_dim, hidden_dim=64, output_dim=1, n_layer=2)
        attack_optimizer = init_optimizer(optimizer_name=args.optimizer, model=attack_model, lr=args.att_lr)

        attack_model = train_attack(args=args, tr_loader=tr_loader, va_loader=va_loader, te_loader=te_loader,
                                    attack_model=attack_model, epochs=args.att_epochs, optimizer=attack_optimizer,
                                    name=name['att'], device=device, history=att_hist)

    attack_model.load_state_dict(torch.load(args.save_path + f"{name['att']}_attack.pt"))

    metric = ['auc', 'acc', 'pre', 'rec', 'f1']
    metric_dict = {
        'auc': torchmetrics.classification.BinaryAUROC().to(device),
        'acc': torchmetrics.classification.BinaryAccuracy().to(device),
        'pre': torchmetrics.classification.BinaryPrecision().to(device),
        'rec': torchmetrics.classification.BinaryRecall().to(device),
        'f1': torchmetrics.classification.BinaryF1Score().to(device)
    }
    for met in metric:
        te_loss, te_auc = eval_attack_step(model=attack_model, device=device, loader=te_loader,
                                       metrics=metric_dict[met], criterion=torch.nn.BCELoss())
        rprint(f"Attack {met}: {te_auc}")
