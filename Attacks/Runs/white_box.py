import os
import sys
import torch
import wandb
import random
import torchmetrics
from loguru import logger
from rich import print as rprint
from Utils.utils import timeit
from Models.models import WbAttacker
from Models.init import init_optimizer
from Attacks.Utils.utils import save_dict
from Attacks.Utils.dataset import ShadowData, custom_collate
from Attacks.Utils.train_eval import train_wb_attack, retrain
from functools import partial

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")

def run(args, graph, model, device, history, name):

    train_g, val_g, test_g, shadow_graph = graph
    model_hist, att_hist = history

    with timeit(logger=logger, task='init-target-model'):

        if args.exist_model == False:
            rprint(f"Model is {args.exist_model} to exist, need to retrain")
            model_name = f"{name['model']}.pt"
            model, model_hist = retrain(args=args, train_g=train_g, val_g=val_g, test_g=test_g, model=model, 
                                        device=device, history=model_hist, name=model_name[:-3])      
            target_model_name = f"{name['model']}.pkl"
            target_model_path = args.res_path + target_model_name
            save_dict(path=target_model_path, dct=model_hist)
        
    with timeit(logger=logger, task='preparing-shadow-data'):
        shadow_graph = shadow_graph.to(device)
        if args.att_submode == 'drop':
            shtr_dataset = ShadowData(graph=shadow_graph, model=model, num_layer=args.n_layers, device=device, mode='train', weight='drop', nnei=4)
            shte_dataset = ShadowData(graph=shadow_graph, model=model, num_layer=args.n_layers, device=device, mode='test', weight='drop', nnei=4)
        else:
            shtr_dataset = ShadowData(graph=shadow_graph, model=model, num_layer=args.n_layers, device=device, mode='train', nnei=4)
            shte_dataset = ShadowData(graph=shadow_graph, model=model, num_layer=args.n_layers, device=device, mode='test', nnei=4)

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
                                                drop_last=False, shuffle=False)
        
        rprint(f"Out dim: {out_dim}")
        rprint(f"Grad dim: {grad_dim}")
        x, y = next(iter(tr_loader))
        _, label, loss, out_dict, grad_dict = x
        rprint(f"Label size: {label.size()}")
        rprint(f"Loss size: {loss.size()}")
        rprint(f"Membership Label size: {y.size()}")
        for key in out_keys:
            rprint(f"Out dict at key {key} has size: {out_dict[key].size()}")
        for key in model_keys:
            rprint(f"Grad dict at key {key} has size: {grad_dict[key].size()}")
        # sys.exit()
        
    with timeit(logger=logger, task='train-attack-model'):

        manual_seed = random.randint(0, 100)
        torch.manual_seed(manual_seed)
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
    id_dict = {}

    # idx = torch.Tensor([]).to(device)
    # pred = torch.Tensor([]).to(device)
    # mem_lab = torch.Tensor([]).to(device)

    node_dict = {}

    for run in range(5):

        for bi, d in enumerate(te_loader):
            features, target = d
            org_id, label, loss_tensor, out_dict, grad_dict = features
            feat = (label, loss_tensor, out_dict, grad_dict)
            target = target.to(device)
            predictions = att_model(feat)
            predictions = torch.squeeze(predictions, dim=-1)
            predictions = torch.nn.functional.sigmoid(predictions)

            org_id = org_id.detach().tolist()
            predictions = predictions.detach().tolist()
            target = target.detach().tolist()

            for i, key in enumerate(org_id):
                if key in node_dict.keys():
                    node_dict[key]['pred'].append(predictions[i])
                else:
                    node_dict[key] = {
                        'label': target[i],
                        'pred': [predictions[i]]
                    }

    idx = []
    lab = []
    pred = []

    for key in node_dict.keys():

        idx.append(key)
        lab.append(node_dict[key]['label'])
        pred.append(sum(node_dict[key]['pred']) / 10)
    
    idx = torch.Tensor(idx)
    lab = torch.Tensor(lab)
    pred = torch.Tensor(pred)

    pred_round = pred.round()
    indx = torch.logical_not(torch.logical_xor(pred_round.int(), lab.int())).nonzero(as_tuple=True)[0]
    correct_predicted_node = idx[indx].int().tolist()

    for m in metric:
        
        met = metric_dict[m]
        perf = met(pred, lab)
        wandb.summary[f'BEST TEST {m}'] = perf
        rprint(f"Attack {m}: {perf}")

    
    wandb.summary[f'Node Correct / times'] = correct_predicted_node
    return model_hist, att_hist
