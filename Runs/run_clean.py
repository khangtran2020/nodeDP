import dgl
import sys
import torch
import torchmetrics
import wandb
from tqdm import tqdm
from Models.train_eval import EarlyStopping, train_fn, eval_fn, performace_eval
from Utils.utils import get_name, save_res
from dgl.dataloading import NeighborSampler
from Utils.utils import timeit
from loguru import logger

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")


def run(args, tr_info, va_info, te_info, model, optimizer, name, device, history):
    _, tr_loader = tr_info
    va_loader = va_info
    _, te_loader = te_info
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

            tk0.set_postfix(Loss=tr_loss, ACC=tr_acc.item(), Va_Loss=va_loss, Va_ACC=va_acc.item(), Te_Loss=te_loss, Te_ACC=te_acc.item())

            results = {"Target train/loss": tr_loss, 
                       f"Target train/{args.performance_metric}": tr_acc.item(), 
                       "Target val/loss": va_loss, 
                       f"Target val/{args.performance_metric}": va_acc.item(),
                       "Target test/loss": te_loss, 
                       f"Target test/{args.performance_metric}": te_acc.item(),
            }
            
            wandb.log({**results})

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
    wandb.summary[f'BEST TEST {args.performance_metric}'] = te_acc
    wandb.summary[f'BEST TEST LOSS'] = test_loss
    history['best_test'] = te_acc.item()
    save_res(name=name, args=args, dct=history)
    return model, history
