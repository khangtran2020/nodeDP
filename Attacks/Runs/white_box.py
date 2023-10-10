import os
import sys
import torch
import torchmetrics
from loguru import logger
from hashlib import md5
from rich import print as rprint
from rich.pretty import pretty_repr
from Utils.utils import timeit, get_index_by_value
from Models.models import WbAttacker
from Models.init import init_model, init_optimizer
from Attacks.Utils.data_utils import generate_nohop_graph
from Attacks.Utils.utils import save_dict
from Attacks.Utils.dataset import Data, ShadowData, custom_collate
from Attacks.Utils.train_eval import train_wb_attack, eval_att_wb_step, retrain, get_entropy
from functools import partial

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")

def run(args, graph, model, device, history, name):

    train_g, val_g, test_g, shadow_graph = graph
    model_hist, att_hist = history

    with timeit(logger=logger, task='init-target-model'):

        if args.exist_model == False:
            rprint(f"Model is {args.exist_model} to exist, need to retrain")
            model_name = f"{md5(name['model'].encode()).hexdigest()}.pt"
            model, model_hist = retrain(args=args, train_g=train_g, val_g=val_g, test_g=test_g, model=model, 
                                        device=device, history=model_hist, name=model_name[:-3])      
            target_model_name = f"{md5(name['model'].encode()).hexdigest()}.pkl"
            target_model_path = args.res_path + target_model_name
            save_dict(path=target_model_path, dct=model_hist)
        
    with timeit(logger=logger, task='preparing-shadow-data'):
        shadow_graph = shadow_graph.to(device)
        if args.att_submode == 'drop':

            with torch.no_grad():
                model.to(device)
                shadow_nohop = generate_nohop_graph(graph=shadow_graph, device=device)
                pred = model.full(g=shadow_nohop, x=shadow_nohop.ndata['feat'])
                conf = get_entropy(pred=pred)
                src_edge, dst_edge = shadow_graph.edges()
                src_conf = conf[src_edge]
                dst_conf = conf[dst_edge]
                conf_exp = torch.exp(-1*torch.abs(src_conf - dst_conf))
                sample = torch.zeros_like(conf_exp).float()
                for node in dst_edge.unique():
                    index = get_index_by_value(a=dst_edge, val=node)
                    conf_exp[index] = conf_exp[index] / (conf_exp[index].sum() + 1e-12)
                    # rprint(f"Median for node {node}: {conf_exp[index] > conf_exp[index].median()}")
                    sample[index] = (conf_exp[index] > conf_exp[index].median()).float()
                shadow_graph.edata['weight'] = conf_exp.to(device)
                shadow_graph.edata['sample'] = sample.to(device)

            shtr_dataset = ShadowData(graph=shadow_graph, model=model, num_layer=args.n_layers, device=device, mode='train', weight='weight', nnei=4)
            shte_dataset = ShadowData(graph=shadow_graph, model=model, num_layer=args.n_layers, device=device, mode='test', weight='sample', nnei=4)
        else:
            shtr_dataset = ShadowData(graph=shadow_graph, model=model, num_layer=args.n_layers, device=device, mode='train')
            shte_dataset = ShadowData(graph=shadow_graph, model=model, num_layer=args.n_layers, device=device, mode='test')

        label, weight = shtr_dataset.label_weight
        lab_weight = 1 - weight / weight.sum()
        rprint(f"Label weight will be: {lab_weight}")

        out_keys = [f'out_{i}' for i in range(args.n_layers)]
        out_dim = []
        if args.n_layers > 1:
            out_dim.append(args.hid_dim)
            for i in range(0, args.n_layers - 2):
                out_dim.append(args.hid_dim)
            out_dim.append(args.num_class)
        else:
            out_dim.append(args.num_class)

        model_keys = []
        grad_dim = []
        for named, p in model.named_parameters():
            if p.requires_grad:
                model_keys.append(named.replace('.', '-'))
                if 'bias' in named:
                    out_d = list(p.size())[0]
                    grad_dim.append((1, out_d))
                else:
                    out_d, in_d = list(p.size())
                    grad_dim.append((in_d, out_d))
                rprint(f"Model parameter {named} has size: {p.size()}")
        
        collate_fn = partial(custom_collate, out_key=out_keys, model_key=model_keys, device=device, num_class=args.num_class)
        tr_loader = torch.utils.data.DataLoader(shtr_dataset, batch_size=args.att_bs, collate_fn=collate_fn,
                                                drop_last=True, shuffle=True)
        te_loader = torch.utils.data.DataLoader(shte_dataset, batch_size=args.att_bs, collate_fn=collate_fn,
                                                drop_last=False, shuffle=True)
        
        rprint(f"Out dim: {out_dim}")
        rprint(f"Grad dim: {grad_dim}")
        x, y = next(iter(tr_loader))
        label, loss, out_dict, grad_dict = x
        rprint(f"Label size: {label.size()}")
        rprint(f"Loss size: {loss.size()}")
        rprint(f"Membership Label size: {y.size()}")
        for key in out_keys:
            rprint(f"Out dict at key {key} has size: {out_dict[key].size()}")
        for key in model_keys:
            rprint(f"Grad dict at key {key} has size: {grad_dict[key].size()}")
        # sys.exit()
        
    with timeit(logger=logger, task='train-attack-model'):
        
        att_model = WbAttacker(label_dim=args.num_class, loss_dim=1, out_dim_list=out_dim, grad_dim_list=grad_dim, 
                               out_keys=out_keys, model_keys=model_keys, num_filters=4, device=device)
        att_model.to(device)
        att_opt = init_optimizer(optimizer_name=args.optimizer, model=att_model, lr=args.att_lr, weight_decay=0)
        att_model = train_wb_attack(args=args, tr_loader=tr_loader, te_loader=te_loader, weight=lab_weight, 
                                    attack_model=att_model, epochs=args.att_epochs, optimizer=att_opt,
                                    name=name['att'], device=device, history=att_hist)

    rprint("\n\n============== BEGIN EVALUATING ==============")
    att_model.load_state_dict(torch.load(args.save_path + f"{name['att']}_attack.pt"))
    # rprint(f"Attack model: {att_model}")
    metric = ['auc', 'acc', 'pre', 'rec', 'f1']
    metric_dict = {
        'auc': torchmetrics.classification.BinaryAUROC().to(device),
        'acc': torchmetrics.classification.BinaryAccuracy().to(device),
        'pre': torchmetrics.classification.BinaryPrecision().to(device),
        'rec': torchmetrics.classification.BinaryRecall().to(device),
        'f1': torchmetrics.classification.BinaryF1Score().to(device)
    }
    for met in metric:
        te_loss, te_auc = eval_att_wb_step(model=att_model, device=device, loader=te_loader, 
                                           metrics=metric_dict[met], criterion=torch.nn.BCELoss())
        rprint(f"Attack {met}: {te_auc}")
    
    return model_hist, att_hist
