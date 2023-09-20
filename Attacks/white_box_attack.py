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
        x, y = generate_attack_samples_white_box(graph=graph, device=device)

        feature = graph.ndata['feat']
        label = graph.ndata['label']
        pred = tar_model.full(g = graph, x = feature)
        loss = criter(pred, label)
        
        pred_loss = torch.nn.functional.sigmoid(loss[x].detach())
        pred_grad = []
        for idx in x:
            grad_norm = 0
            loss[idx].backward(retain_graph=True)
            for tensor_name, tensor in tar_model.named_parameters():
                if tensor.grad is not None:
                    grad_norm = grad_norm + tensor.grad.detach().norm(p=2)**2
            pred_grad.append(torch.nn.functional.sigmoid(grad_norm).item())
        pred_grad = torch.Tensor(pred_grad)

        auc_loss = metrics(pred_loss, y)
        auc_grad = metrics(pred_grad, y)
        rprint(f"Attack AUC on loss: {auc_loss.item()}, on grad: {auc_grad.item()}")    