import torch
import torchmetrics
from rich import print as rprint
from rich.pretty import pretty_repr
from tqdm import tqdm
from Models.train_eval import EarlyStopping, train_mlp, eval_mlp
from Utils.utils import save_res


def run(args, tr_info, va_info, te_info, model, optimizer, name, device, history):
    
    print(f'Data has {args.num_feat} features and {args.num_class} classes')
    graph, tr_loader = tr_info
    va_loader = va_info
    _, te_loader = te_info
    model_name = '{}.pt'.format(name)
    mode = args.mlp_mode

    criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    criter = torch.nn.CrossEntropyLoss(reduction='none')
    criter.to(device)

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

    es = EarlyStopping(patience=args.patience, verbose=False)

    tk0 = tqdm(range(args.epochs), total=args.epochs, position=0, colour='green',
               bar_format='{l_bar}{bar:20}{r_bar}{bar:-20b}')
    for epoch in tk0:
        if mode == 'clean':
            criter_tr = criterion
        else:
            criter_tr = criter
        tr_loss, tr_acc = train_mlp(loader=tr_loader, model=model, criter=criter_tr, optimizer=optimizer, device=device, metrics=metrics, mode=mode, clip=args.clip, ns=args)
        # scheduler.step()
        va_loss, va_acc = eval_mlp(loader=va_loader, model=model, criter=criterion,
                                  metrics=metrics, device=device)
        te_loss, te_acc = eval_mlp(loader=te_loader, model=model, criter=criterion,
                                  metrics=metrics, device=device)

        # scheduler.step(va_loss)

        tk0.set_postfix(Loss=tr_loss, ACC=tr_acc, Va_Loss=va_loss, Va_ACC=va_acc, Te_ACC=te_acc)

        history['train_history_loss'].append(tr_loss)
        history['train_history_acc'].append(tr_acc)
        history['val_history_loss'].append(va_loss)
        history['val_history_acc'].append(va_acc)
        history['test_history_loss'].append(te_loss)
        history['test_history_acc'].append(te_acc)
        es(epoch=epoch, epoch_score=va_acc, model=model, model_path=args.save_path + model_name)
        # torch.save(model.state_dict(), args.save_path + model_name)

    model.load_state_dict(torch.load(args.save_path + model_name))
    test_loss, te_acc = eval_mlp(loader=te_loader, model=model, criter=criterion,
                                  metrics=metrics, device=device)
    history['best_test'] = te_acc
    if args.debug:
        rprint(pretty_repr(history))
    save_res(name=name, args=args, dct=history)

    return model, history