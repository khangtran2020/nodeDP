import sys
import torch
import torchmetrics
from loguru import logger
from hashlib import md5
from rich import print as rprint
from rich.pretty import pretty_repr
from Utils.utils import timeit
from Models.models import WbAttacker
from Models.init import init_model, init_optimizer
from Attacks.Utils.utils import save_dict
from Attacks.Utils.dataset import Data, ShadowData, custom_collate
from Attacks.Utils.train_eval import train_shadow, train_attack, eval_attack_step, retrain
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
        
        shtr_dataset = ShadowData(graph=shadow_graph, model=model, num_layer=args.n_layers, device=device, mode='train')
        shte_dataset = ShadowData(graph=shadow_graph, model=model, num_layer=args.n_layers, device=device, mode='test')

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
        # x, y = next(iter(tr_loader))
        # label, loss, out_dict, grad_dict = x
        # rprint(f"Label size: {label.size()}")
        # rprint(f"Loss size: {loss.size()}")
        # rprint(f"Membership Label size: {y.size()}")
        # for key in out_keys:
        #     rprint(f"Out dict at key {key} has size: {out_dict[key].size()}")
        # for key in model_keys:
        #     rprint(f"Grad dict at key {key} has size: {grad_dict[key].size()}")
        # sys.exit()
        

    # device = torch.device('cpu')
    with timeit(logger=logger, task='train-attack-model'):
        
        att_model = WbAttacker(label_dim=args.num_class, loss_dim=1, out_dim_list=out_dim, grad_dim_list=grad_dim, 
                               out_keys=out_keys, model_keys=model_keys, num_filters=4, device=device)
        att_model.to(device)
        att_opt = init_optimizer(optimizer_name=args.optimizer, model=att_model, lr=args.att_lr)

        x, y = next(iter(tr_loader))
        pred = att_model(x)
        rprint(f"Prediction: {pred}, with size {pred.size()}")
        sys.exit()

    #     attack_model = NN(input_dim=new_dim, hidden_dim=64, output_dim=1, n_layer=3)
    #     attack_optimizer = init_optimizer(optimizer_name=args.optimizer, model=attack_model, lr=args.lr)

    #     attack_model = train_attack(args=args, tr_loader=tr_loader, va_loader=va_loader, te_loader=te_loader,
    #                                 attack_model=attack_model, epochs=args.attack_epochs, optimizer=attack_optimizer,
    #                                 name=tar_history['name'], device=device)

    # attack_model.load_state_dict(torch.load(args.save_path + f"{tar_history['name']}_attack.pt"))
    # te_loss, te_auc = eval_attack_step(model=attack_model, device=device, loader=te_loader,
    #                                    metrics=torchmetrics.classification.BinaryAUROC().to(device),
    #                                    criterion=torch.nn.BCELoss())
    # rprint(f"Attack AUC: {te_auc}")
