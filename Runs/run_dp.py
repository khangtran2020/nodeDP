import dgl
import torch

from tqdm import tqdm
from Models.train_eval import EarlyStopping, train_dp, eval_fn, performace_eval
from Utils.utils import save_res


def run(args, dataloaders, model, optimizer, name, device, graph, logger):
    print(f'Data has {args.num_feat} features and {args.num_class} classes')
    tr_loader, val_loader, te_loader = dataloaders
    model_name = '{}.pt'.format(name)
    model.to(device)
    # DEfining criterion
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)

    # DEfining Early Stopping Object
    es = EarlyStopping(patience=args.patience, verbose=False)

    # History dictionary to store everything
    history = {
        'train_history_loss': [],
        'train_history_acc': [],
        'val_history_loss': [],
        'val_history_acc': [],
        'test_history_loss': [],
        'test_history_acc': [],
        'best_test': 0
    }

    # THE ENGINE LOOP
    tk0 = tqdm(range(args.epochs), total=args.epochs)
    for epoch in tk0:
        train_loss, train_out, train_targets = train_dp(dataloader=tr_loader, model=model, criterion=criterion,
                                                        optimizer=optimizer, device=device, scheduler=None, g=graph,
                                                        clip_grad=args.clip, clip_node=args.clip_node, ns=args.ns,
                                                        trim_rule=args.trim_rule)
        val_loss, val_outputs, val_targets = eval_fn(data_loader=val_loader, model=model, criterion=criterion,
                                                     device=device)
        test_loss, test_outputs, test_targets = eval_fn(data_loader=te_loader, model=model, criterion=criterion,
                                                        device=device)

        train_acc = performace_eval(args, train_targets, train_out)
        test_acc = performace_eval(args, test_targets, test_outputs)
        val_acc = performace_eval(args, val_targets, val_outputs)

        # scheduler.step(acc_score)

        tk0.set_postfix(Train_Loss=train_loss, Train_SCORE=train_acc, Valid_Loss=val_loss,
                        Valid_SCORE=val_acc)

        history['train_history_loss'].append(train_loss)
        history['train_history_acc'].append(train_acc)
        history['val_history_loss'].append(val_loss)
        history['val_history_acc'].append(val_acc)
        history['test_history_loss'].append(test_loss)
        history['test_history_acc'].append(test_acc)
        es(epoch=epoch, epoch_score=val_acc, model=model, model_path=args.save_path + model_name)
        # torch.save(model.state_dict(), args.save_path + model_name)

    model.load_state_dict(torch.load(args.save_path + model_name))
    test_loss, test_outputs, test_targets = eval_fn(te_loader, model, criterion, device)
    test_acc = performace_eval(args, test_targets, test_outputs)
    history['best_test'] = test_acc
    save_res(name=name, args=args, dct=history)
