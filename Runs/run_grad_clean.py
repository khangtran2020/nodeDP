import sys
import time
import torch
import torchmetrics
from copy import deepcopy
from rich import print as rprint
from rich.pretty import pretty_repr
from tqdm import tqdm
from Models.train_eval import EarlyStopping, train_fn_grad_inspect, eval_grad_inspect_fn
from Utils.utils import save_res, timeit
from loguru import logger

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")

# args, tr_info, va_info, te_info, model, optimizer, name, device

def run(args, tr_info, va_info, te_info, model, optimizer, name, device, history):

    print(f'Data has {args.num_feat} features and {args.num_class} classes')

    _, tr_loader = tr_info
    va_loader = va_info
    _, te_loader = te_info

    model_name = '{}.pt'.format(name)

    model_clean = deepcopy(model)
    model.to(device)
    model_clean.to(device)
    # DEfining criterion

    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)

    if args.performance_metric == 'acc':
        metrics = torchmetrics.classification.Accuracy(task="multiclass", num_classes=args.num_class).to(device)
    elif args.performance_metric == 'pre':
        metrics = torchmetrics.classification.Precision(task="multiclass", num_classes=args.num_class).to(device)
    elif args.performance_metric == 'f1':
        metrics = torchmetrics.classification.F1Score(task="multiclass", num_classes=args.num_class).to(device)
    elif args.performance_metric == 'auc':
        metrics = torchmetrics.classification.AUROC(task="multiclass", num_classes=args.num_class).to(device)
    else:
        metrics = None

    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, min_lr=1e-3, factor=0.9, patience=30)

    # DEfining Early Stopping Object
    es = EarlyStopping(patience=args.patience, verbose=False)

    # THE ENGINE LOOP
    tk0 = tqdm(range(args.epochs), total=args.epochs, position=0, colour='green',
               bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')

    
    with timeit(logger=logger, task="training-process"):

        for epoch in tk0:
            states = model.state_dict()
            model_clean.load_state_dict(states)
            t0 = time.time()
            tr_loss, tr_acc, grad_ntr, grad_vtr = train_fn_grad_inspect(args=args, dataloader=tr_loader, model=model, 
                                                                criterion=criterion, optimizer=optimizer, device=device, 
                                                                metric=metrics, scheduler=None)
            t1 = time.time()
            t = t1 - t0

            # scheduler.step()
            va_loss, va_acc, grad_vva, grad_nva = eval_grad_inspect_fn(data_loader=va_loader, model=model, criterion=criterion,
                                    metric=metrics, device=device)
            te_loss, te_acc, grad_vte, grad_nte = eval_grad_inspect_fn(data_loader=te_loader, model=model, criterion=criterion,
                                    metric=metrics, device=device)

            cons = (grad_vtr*grad_vte).sum().item() / (grad_vtr.norm(p=2).item() * grad_vte.norm(p=2).item() + 1e-12)

            rprint(f"At epoch {epoch}: norm of avg grad in train {grad_ntr}, norm of avg grad in test {grad_nte}, cosine between two vec: {cons}")

            tk0.set_postfix(Loss=tr_loss, ACC=tr_acc.item(), Va_Loss=va_loss, Va_ACC=va_acc.item(), Te_ACC=te_acc.item())

            history['train_history_loss'].append(tr_loss)
            history['train_history_acc'].append(tr_acc.item())
            history['val_history_loss'].append(va_loss)
            history['val_history_acc'].append(va_acc.item())
            history['test_history_loss'].append(te_loss)
            history['test_history_acc'].append(te_acc.item())
            es(epoch=epoch, epoch_score=va_acc.item(), model=model, model_path=args.save_path + model_name)
            # torch.save(model.state_dict(), args.save_path + model_name)

    model.load_state_dict(torch.load(args.save_path + model_name))
    test_loss, te_acc = eval_fn(te_loader, model, criterion, metric=metrics, device=device)
    history['best_test'] = te_acc.item()
    if args.debug:
        rprint(pretty_repr(history))
    save_res(name=name, args=args, dct=history)

    return model, history
