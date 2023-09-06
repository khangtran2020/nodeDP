import dgl
import sys
import torch
import torchmetrics

from tqdm import tqdm
from Models.train_eval import EarlyStopping, train_fn, eval_fn, performace_eval
from Utils.utils import get_name, save_res
from dgl.dataloading import NeighborSampler
from Utils.utils import timeit
from loguru import logger
from Analysis.Struct.read import read_data
from rich import print as rprint
from Data.read import init_loader

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")


def run(args, model, optimizer, name, device, history):

    with timeit(logger, 'init-data'):
        
        org_info, mod_info = read_data(args=args, data_name=args.dataset, history=history)
        
        tr_g, va_g, te_g = org_info
        tr_g_, va_g_, te_g_ = mod_info

        tr_g = tr_g.to(device)
        va_g = va_g.to(device)
        te_g = te_g.to(device)

        tr_g_ = tr_g_.to(device)
        va_g_ = va_g_.to(device)
        te_g_ = te_g_.to(device)

        tr_loader, va_loader, te_loader = init_loader(args=args, device=device, train_g=tr_g, test_g=te_g, val_g=va_g)
        tr_loader_, va_loader_, te_loader_ = init_loader(args=args, device=device, train_g=tr_g_, test_g=te_g_, val_g=va_g_)

    
    model_name = '{}.pt'.format(name)
    model.to(device)

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

    # DEfining Early Stopping Object
    es = EarlyStopping(patience=args.patience, verbose=False)

    with timeit(logger=logger, task="training-process"):
        # THE ENGINE LOOP
        tk0 = tqdm(range(args.epochs), total=args.epochs)
        for epoch in tk0:
            tr_loss, tr_acc = train_fn(dataloader=tr_loader, model=model, criterion=criterion,
                                    optimizer=optimizer, device=device, scheduler=None, metric=metrics)
            va_loss, va_acc = eval_fn(data_loader=va_loader, model=model, criterion=criterion,
                                    device=device, metric=metrics)
            te_loss, te_acc = eval_fn(data_loader=te_loader, model=model, criterion=criterion,
                                    device=device, metric=metrics)

            # scheduler.step(acc_score)

            tk0.set_postfix(Loss=tr_loss, ACC=tr_acc.item(), Va_Loss=va_loss, Va_ACC=va_acc.item(), Te_ACC=te_acc.item())

            history['train_history_loss'].append(tr_loss)
            history['train_history_acc'].append(tr_acc.item())
            history['val_history_loss'].append(va_loss)
            history['val_history_acc'].append(va_acc.item())
            history['test_history_loss'].append(te_loss)
            history['test_history_acc'].append(te_acc.item())
            es(epoch=epoch, epoch_score=va_acc.item(), model=model, model_path=args.save_path + model_name)
            # if es.early_stop:
            #     break

    model.load_state_dict(torch.load(args.save_path + model_name))
    test_loss, te_acc = eval_fn(te_loader, model, criterion, metric=metrics, device=device)
    history['best_test'] = te_acc.item()
    save_res(name=name, args=args, dct=history)
    return model, history
