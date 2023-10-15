import sys
import dgl
import torch
import wandb
import torchmetrics
from functools import partial
from loguru import logger
from rich.pretty import pretty_repr
from rich import print as rprint
from Utils.console import console
from Utils.utils import get_index_by_value
from Utils.utils import timeit
from Attacks.Utils.utils import save_dict
from Attacks.Utils.data_utils import generate_nohop_graph
from Attacks.Utils.train_eval import retrain, train_link_attack, train_wb_attack, eval_att_wb_step
from Attacks.Utils.dataset import ShadowLinkData, ShadowData, custom_collate
from Models.models import LinkNN, WbAttacker
from Models.init import init_optimizer

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
        
    with timeit(logger=logger, task='train-link-model'):
        
        shadow_graph = shadow_graph.to(device)
        shadow_graph_nohop = generate_nohop_graph(graph=shadow_graph, device=device)

        src_edge, dst_edge = shadow_graph.edges()

        # get edges in shadow train & test
        str_mask = shadow_graph.ndata['str_mask']
        ste_mask = shadow_graph.ndata['ste_mask']

        src_edge_intr = str_mask[src_edge]
        dst_edge_intr = str_mask[dst_edge]
        mask_intr = torch.logical_and(src_edge_intr, dst_edge_intr).int()
        idx_edge_intr = get_index_by_value(a=mask_intr, val=1)

        src_edge_inte = ste_mask[src_edge]
        dst_edge_inte = ste_mask[dst_edge]
        mask_inte = torch.logical_and(src_edge_inte, dst_edge_inte).int()
        idx_edge_inte = get_index_by_value(a=mask_inte, val=1)

        console.log(f"Index in train & test: {torch.isin(idx_edge_intr, idx_edge_inte).sum().item() + torch.isin(idx_edge_inte, idx_edge_intr).sum().item()}")

        # get edges in the same set in train 
        pos_mask_tr = shadow_graph.ndata['pos_mask_tr']
        neg_mask_tr = shadow_graph.ndata['neg_mask_tr']

        src_edge_pos_intr = pos_mask_tr[src_edge]
        dst_edge_pos_intr = pos_mask_tr[dst_edge]
        mask_pos_intr = torch.logical_and(src_edge_pos_intr, dst_edge_pos_intr).int()
        indx_pos_intr = get_index_by_value(a=mask_pos_intr, val=1)

        src_edge_neg_intr = neg_mask_tr[src_edge]
        dst_edge_neg_intr = neg_mask_tr[dst_edge]
        mask_neg_intr = torch.logical_and(src_edge_neg_intr, dst_edge_neg_intr).int()
        indx_neg_intr = get_index_by_value(a=mask_neg_intr, val=1)

        indx_same_intr = torch.cat((indx_pos_intr, indx_neg_intr), dim=0)

        console.log(f"Index in train pos & train neg: {torch.isin(indx_pos_intr, indx_neg_intr).sum().item() + torch.isin(indx_neg_intr, indx_pos_intr).sum().item()}")
        console.log(f"Index in train same in train: {torch.isin(indx_same_intr, idx_edge_intr).sum().item() - indx_same_intr.size(dim=0)}")

        # get edges in diff set in train
        mask_pos_neg_intr = torch.logical_and(src_edge_pos_intr, dst_edge_neg_intr).int()
        indx_pos_neg_intr = get_index_by_value(a=mask_pos_neg_intr, val=1)

        mask_neg_pos_intr = torch.logical_and(src_edge_neg_intr, dst_edge_pos_intr).int()
        indx_neg_pos_intr = get_index_by_value(a=mask_neg_pos_intr, val=1)

        indx_diff_intr = torch.cat((indx_pos_neg_intr, indx_neg_pos_intr), dim=0)

        console.log(f"Index in train pos neg & train neg pos: {torch.isin(indx_pos_neg_intr, indx_neg_pos_intr).sum().item() + torch.isin(indx_neg_pos_intr, indx_pos_neg_intr).sum().item()}")
        console.log(f"Index in train diff in train: {torch.isin(indx_diff_intr, idx_edge_intr).sum().item() - indx_diff_intr.size(dim=0)}")
        
        # get edges in the same set in test 
        pos_mask_te = shadow_graph.ndata['pos_mask_te']
        neg_mask_te = shadow_graph.ndata['neg_mask_te']

        src_edge_pos_inte = pos_mask_te[src_edge]
        dst_edge_pos_inte = pos_mask_te[dst_edge]
        mask_pos_inte = torch.logical_and(src_edge_pos_inte, dst_edge_pos_inte).int()
        indx_pos_inte = get_index_by_value(a=mask_pos_inte, val=1)

        src_edge_neg_inte = neg_mask_te[src_edge]
        dst_edge_neg_inte = neg_mask_te[dst_edge]
        mask_neg_inte = torch.logical_and(src_edge_neg_inte, dst_edge_neg_inte).int()
        indx_neg_inte = get_index_by_value(a=mask_neg_inte, val=1)

        indx_same_inte = torch.cat((indx_pos_inte, indx_neg_inte), dim=0)

        console.log(f"Index in test pos & test neg: {torch.isin(indx_pos_inte, indx_neg_inte).sum().item() + torch.isin(indx_neg_inte, indx_pos_inte).sum().item()}")
        console.log(f"Index in test same in test: {torch.isin(indx_same_inte, idx_edge_inte).sum().item() - indx_same_inte.size(dim=0)}")

        # get edges in diff set in test
        mask_pos_neg_inte = torch.logical_and(src_edge_pos_inte, dst_edge_neg_inte).int()
        indx_pos_neg_inte = get_index_by_value(a=mask_pos_neg_inte, val=1)

        mask_neg_pos_inte = torch.logical_and(src_edge_neg_inte, dst_edge_pos_inte).int()
        indx_neg_pos_inte = get_index_by_value(a=mask_neg_pos_inte, val=1)

        indx_diff_inte = torch.cat((indx_pos_neg_inte, indx_neg_pos_inte), dim=0)

        console.log(f"Index in test pos neg & test neg pos: {torch.isin(indx_pos_neg_inte, indx_neg_pos_inte).sum().item() + torch.isin(indx_neg_pos_inte, indx_pos_neg_inte).sum().item()}")
        console.log(f"Index in test diff in test: {torch.isin(indx_diff_inte, idx_edge_inte).sum().item() - indx_diff_inte.size(dim=0)}")
        
        edge_dict = {
            '# train pos': indx_same_intr.size(dim=0), 
            '# train neg': indx_diff_intr.size(dim=0),
            '# test pos': indx_same_inte.size(dim=0),
            '# test neg': indx_diff_inte.size(dim=0)
        }
        console.log(f"Edge info: {pretty_repr(edge_dict)}")
        
        idx_tr = torch.cat((indx_same_intr, indx_diff_intr), dim=0)
        idx_te = torch.cat((indx_same_inte, indx_diff_inte), dim=0)

        y_tr = torch.cat((torch.ones(indx_same_intr.size(dim=0)), torch.zeros(indx_diff_intr.size(dim=0))), dim=0)
        y_te = torch.cat((torch.ones(indx_same_inte.size(dim=0)), torch.zeros(indx_diff_inte.size(dim=0))), dim=0)


        src_tr = src_edge[idx_tr]
        dst_tr = dst_edge[idx_tr]

        src_te = src_edge[idx_te]
        dst_te = dst_edge[idx_te]

        edge_list_tr = list(zip(src_tr.cpu().tolist(), dst_tr.cpu().tolist()))
        edge_list_te = list(zip(src_te.cpu().tolist(), dst_te.cpu().tolist()))

        linktr_dataset = ShadowLinkData(edge_list=edge_list_tr, label=y_tr, graph=shadow_graph_nohop, model=model, device=device)
        linkte_dataset = ShadowLinkData(edge_list=edge_list_te, label=y_te, graph=shadow_graph_nohop, model=model, device=device)

        linktr_loader = torch.utils.data.DataLoader(linktr_dataset, batch_size=128, drop_last=True, shuffle=True)
        linkte_loader = torch.utils.data.DataLoader(linkte_dataset, batch_size=125, drop_last=False, shuffle=False)


        link_model = LinkNN(input_dim=linktr_dataset.num_feat, hidden_dim=64, output_dim=1, n_layer=3, dropout=0.2)
        link_model.to(device)
        link_opt = init_optimizer(optimizer_name=args.optimizer, model=link_model, lr=0.001, weight_decay=1e-4)
        link_model = train_link_attack(args=args, tr_loader=linktr_loader, te_loader=linkte_loader,
                                    link_model=link_model, epochs=200, optimizer=link_opt,
                                    name=name['att'], device=device, history=att_hist)
        
        link_model.load_state_dict(torch.load(args.save_path + f"{name['att']}_link.pt"))
        edge_pred = torch.Tensor([]).to(device)
        with torch.no_grad():
            for bi, d in enumerate(linkte_loader):
                features, target = d
                x1, x2 = features
                predictions = link_model(x1.to(device), x2.to(device))
                predictions = torch.squeeze(predictions, dim=-1)
                edge_pred = torch.cat((edge_pred, predictions.round()), dim=0)
        
        idx = get_index_by_value(a=edge_pred.int(), val=1)

        src_te = src_te[idx]
        dst_te = dst_te[idx]

        src_tr = src_edge[indx_same_intr]
        dst_tr = dst_edge[indx_same_intr]        

        src_edge = torch.cat((src_tr, src_te), dim=0)
        dst_edge = torch.cat((dst_tr, dst_te), dim=0)

        shadow_graph_rep = dgl.graph((src_edge, dst_edge), num_nodes=shadow_graph.nodes().size(dim=0)).to(device)
        for key in shadow_graph.ndata.keys():
            shadow_graph_rep.ndata[key] = shadow_graph.ndata[key].clone()
        
    with timeit(logger=logger, task='preparing-shadow-data'):

        shadow_graph_rep = shadow_graph_rep.to(device)
        if args.att_submode == 'drop':
            shtr_dataset = ShadowData(graph=shadow_graph_rep, model=model, num_layer=args.n_layers, device=device, mode='train', weight='drop', nnei=4)
            shte_dataset = ShadowData(graph=shadow_graph_rep, model=model, num_layer=args.n_layers, device=device, mode='test', weight='drop', nnei=4)
        else:
            shtr_dataset = ShadowData(graph=shadow_graph_rep, model=model, num_layer=args.n_layers, device=device, mode='train', nnei=4)
            shte_dataset = ShadowData(graph=shadow_graph_rep, model=model, num_layer=args.n_layers, device=device, mode='test', nnei=4)

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
        
        att_model = WbAttacker(label_dim=args.num_class, loss_dim=1, out_dim_list=out_dim, grad_dim_list=grad_dim, 
                               out_keys=out_keys, model_keys=model_keys, num_filters=4, device=device)
        att_model.to(device)
        att_opt = init_optimizer(optimizer_name=args.optimizer, model=att_model, lr=args.att_lr, weight_decay=0)
        att_model = train_wb_attack(args=args, tr_loader=tr_loader, te_loader=te_loader, weight=lab_weight, 
                                    attack_model=att_model, epochs=args.att_epochs, optimizer=att_opt,
                                    name=name['att'], device=device, history=att_hist)

    # rprint("\n\n============== BEGIN EVALUATING ==============")
    # att_model.load_state_dict(torch.load(args.save_path + f"{name['att']}_attack.pt"))
    # # rprint(f"Attack model: {att_model}")
    # metric = ['auc', 'acc', 'pre', 'rec', 'f1']
    # metric_dict = {
    #     'auc': torchmetrics.classification.BinaryAUROC().to(device),
    #     'acc': torchmetrics.classification.BinaryAccuracy().to(device),
    #     'pre': torchmetrics.classification.BinaryPrecision().to(device),
    #     'rec': torchmetrics.classification.BinaryRecall().to(device),
    #     'f1': torchmetrics.classification.BinaryF1Score().to(device)
    # }
    # id_dict = {}

    # for met in metric:
    #     te_loss, te_auc, org_id  = eval_att_wb_step(model=att_model, device=device, loader=te_loader, 
    #                                        metrics=metric_dict[met], criterion=torch.nn.BCELoss(), mode='best')
        
    #     for i in org_id:
    #         if f"{int(i)}" in id_dict.keys():
    #             id_dict[f"{int(i)}"] += 1
    #         else:
    #             id_dict[f"{int(i)}"] = 1

    #     wandb.summary[f'BEST TEST {met}'] = te_auc
    #     rprint(f"Attack {met}: {te_auc}")

    
    # wandb.summary[f'Node Correct / times'] = id_dict
    return model_hist, att_hist