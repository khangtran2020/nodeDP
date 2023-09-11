import dgl
import sys
import torch
import torchmetrics
import networkx as nx
import numpy as np
from tqdm import tqdm
from Models.train_eval import EarlyStopping, train_fn, eval_fn, train_nodedp
from Utils.utils import save_res
from Utils.utils import timeit
from loguru import logger
from Models.init import init_model, init_optimizer
# from Analysis.Struct.read import read_data
from rich import print as rprint
from Data.read import init_loader, read_data
from sklearn.manifold import TSNE

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")


def run(args, name, device, history):

    with timeit(logger, 'init-data'):
        
        g_train, g_val, g_test, _ = read_data(args=args, data_name=args.dataset, history=history)

        with timeit(logger=logger, task='init-val-laplacian'):
            va_adj = g_val.adj_external(scipy_fmt='csr')
            G_va = nx.from_scipy_sparse_matrix(va_adj)
            L_va = nx.laplacian_matrix(G_va).tocoo()
            values = L_va.data
            indices = np.vstack((L_va.row, L_va.col))
            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = L_va.shape
            va_Lsp = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)
            del G_va, L_va, values, indices, i, v, shape, va_adj

        with timeit(logger=logger, task='init-test-laplacian'):
            te_adj = g_test.adj_external(scipy_fmt='csr')
            G_te = nx.from_scipy_sparse_matrix(te_adj)
            L_te = nx.laplacian_matrix(G_te).tocoo()
            values = L_te.data
            indices = np.vstack((L_te.row, L_te.col))
            i = torch.LongTensor(indices)
            v = torch.FloatTensor(values)
            shape = L_te.shape
            te_Lsp = torch.sparse.FloatTensor(i, v, torch.Size(shape)).to(device)
            del G_te, L_te, values, indices, i, v, shape, te_adj



        tr_g = g_train.to(device)
        va_g = g_val.to(device)
        te_g = g_test.to(device)

        tr_loader, va_loader, te_loader = init_loader(args=args, device=device, train_g=tr_g, test_g=te_g, val_g=va_g)
        
        
    model = init_model(args=args)
    optimizer = init_optimizer(optimizer_name=args.optimizer, model=model, lr=args.lr)
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
        step_save = 0
        for epoch in tk0:
            if args.mode == 'clean':
                tr_loss, tr_acc = train_fn(dataloader=tr_loader, model=model, criterion=criterion,
                                        optimizer=optimizer, device=device, scheduler=None, metric=metrics)
            else:
                criter = torch.nn.CrossEntropyLoss(reduction='none').to(device)
                tr_loss, tr_acc = train_nodedp(args=args, dataloader=tr_loader, model=model,
                                                criterion=criter, optimizer=optimizer, device=device,
                                                scheduler=None, g=tr_g, clip_grad=args.clip,
                                                clip_node=args.clip_node, ns=args.ns,
                                                trim_rule=args.trim_rule, history=history, step=epoch,
                                                metric=metrics)
            
            va_loss, va_acc = eval_fn(data_loader=va_loader, model=model, criterion=criterion,
                                    device=device, metric=metrics)
            te_loss, te_acc = eval_fn(data_loader=te_loader, model=model, criterion=criterion,
                                    device=device, metric=metrics)
            with torch.no_grad():
                te_conf = model.full(te_g, te_g.ndata['feat'])
                diff = torch.trace(torch.mm(te_conf.transpose(0, 1), torch.sparse.mm(te_Lsp, te_conf))).sqrt().item()
                history['te_avg_smooth'].append(diff)
                va_conf = model.full(va_g, va_g.ndata['feat'])
                diff = torch.trace(torch.mm(va_conf.transpose(0, 1), torch.sparse.mm(va_Lsp, va_conf))).sqrt().item()
                history['va_avg_smooth'].append(diff)
                del diff

                if epoch % int(args.epochs/4) == 0:
                    step_save += 1
                    t_sne = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3)
                    X_te = te_conf.cpu().numpy()
                    y_te = te_g.ndata['label'].cpu().numpy()
                    X_te_emb = t_sne.fit_transform(X_te, y_te)
                    X_va = va_conf.cpu().numpy()
                    y_va = va_g.ndata['label'].cpu().numpy()
                    X_va_emb = t_sne.fit_transform(X_va, y_va)
                    history[f'tsne_te_step_{step_save}'] = X_te_emb
                    history[f'tsne_va_step_{step_save}'] = X_va_emb

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
    te_conf = model.full(te_g, te_g.ndata['feat'])
    diff = torch.trace(torch.mm(te_conf.transpose(0, 1), torch.sparse.mm(te_Lsp, te_conf))).sqrt().item()
    history['best_te_smooth'] = diff
    va_conf = model.full(va_g, va_g.ndata['feat'])
    diff = torch.trace(torch.mm(va_conf.transpose(0, 1), torch.sparse.mm(va_Lsp, va_conf))).sqrt().item()
    history['best_va_smooth'] = diff
    save_res(name=name, args=args, dct=history)
    return model, history
