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
from Attacks.helper import generate_attack_samples_white_box, Data
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


def run_white_box(args, current_time, device):

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

    with timeit(logger=logger, task='preparing-attack-data'):
        tar_model.to(device)
        graph = graph.to(device)
        tar_model.zero_grad()
        metrics = torchmetrics.classification.AUROC(task="binary").to(device)

        criter = torch.nn.CrossEntropyLoss(reduction='none')
        x_tr, x_te, y_tr, y_te = generate_attack_samples_white_box(graph=graph, device=device)

        feature = graph.ndata['feat']
        label = graph.ndata['label']
        pred = tar_model.full(g = graph, x = feature)
        loss = criter(pred, label)

        x_tr_feat = None
        for i, idx in enumerate(x_tr):
            grad = torch.Tensor([]).to(device)
            loss[idx].backward(retain_graph=True)
            for _, tensor in tar_model.named_parameters():
                if tensor.grad is not None:
                    grad =  torch.cat((grad, torch.flatten(tensor.grad.detach().clone())), dim = 0)
            if i == 0:
                x_tr_feat = torch.unsqueeze(grad, dim=0).clone()
            else:
                x_tr_feat = torch.cat((x_tr_feat, torch.unsqueeze(grad, dim=0)), dim=0)

        x_tr_feat = torch.cat((x_tr_feat, pred[x_tr].detach()), dim=1)
        x_tr_feat = x_tr_feat.to(device)

        x_te_feat = None
        for i, idx in enumerate(x_te):
            grad = torch.Tensor([]).to(device)
            loss[idx].backward(retain_graph=True)
            for _, tensor in tar_model.named_parameters():
                if tensor.grad is not None:
                    grad =  torch.cat((grad, torch.flatten(tensor.grad.detach().clone())), dim = 0)
            if i == 0:
                x_te_feat = torch.unsqueeze(grad, dim=0).clone()
            else:
                x_te_feat = torch.cat((x_te_feat, torch.unsqueeze(grad, dim=0)), dim=0)

        x_te_feat = torch.cat((x_te_feat, pred[x_te].detach()), dim=1)
        x_te_feat = x_te_feat.to(device)

        tr_data = Data(X=x_tr_feat, y=y_tr)
        te_data = Data(X=x_te_feat, y=y_te)

    with timeit(logger=logger, task='train-attack-model'):
        tr_loader = torch.utils.data.DataLoader(tr_data, batch_size=args.batch_size,
                                                pin_memory=False, drop_last=True, shuffle=True)
        te_loader = torch.utils.data.DataLoader(te_data, batch_size=args.batch_size, num_workers=0, shuffle=False,
                                                pin_memory=False, drop_last=False)
        
        attack_model = NN(input_dim=x_tr_feat.size(dim=1), hidden_dim=16, output_dim=1, n_layer=2)
        attack_optimizer = init_optimizer(optimizer_name=args.optimizer, model=attack_model, lr=args.lr)

        attack_model = train_attack(args=args, tr_loader=tr_loader, va_loader=None, te_loader=te_loader,
                                    attack_model=attack_model, epochs=args.attack_epochs, optimizer=attack_optimizer,
                                    name=tar_history['name'], device=device)

    attack_model.load_state_dict(torch.load(args.save_path + f"{tar_history['name']}_attack.pt"))
    te_loss, te_auc = eval_attack_step(model=attack_model, device=device, loader=te_loader,
                                       metrics=torchmetrics.classification.BinaryAUROC().to(device),
                                       criterion=torch.nn.BCELoss())
    rprint(f"Attack AUC: {te_auc}")