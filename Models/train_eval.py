import dgl
import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score
from Trim.base import AppearDict
from copy import deepcopy


class EarlyStopping:
    def __init__(self, patience=7, mode="max", delta=0.001, verbose=False, run_mode=None, skip_ep=100):
        self.patience = patience
        self.counter = 0
        self.mode = mode
        self.best_score = None
        self.best_epoch = 0
        self.early_stop = False
        self.delta = delta
        self.verbose = verbose
        if self.mode == "min":
            self.val_score = np.Inf
        else:
            self.val_score = -np.Inf
        self.run_mode = run_mode
        self.skip_ep = skip_ep

    def __call__(self, epoch, epoch_score, model, model_path):
        score = np.copy(epoch_score)
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(epoch_score, model, model_path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
            self.save_checkpoint(epoch_score, model, model_path)
            self.counter = 0

    def save_checkpoint(self, epoch_score, model, model_path):
        if epoch_score not in [-np.inf, np.inf, -np.nan, np.nan]:
            if self.verbose:
                print('Validation score improved ({} --> {}). Saving model!'.format(self.val_score, epoch_score))
            if self.run_mode != 'func':
                torch.save(model.state_dict(), model_path)
            else:
                torch.save(model, model_path)
        self.val_score = epoch_score


def update_clean(model, optimizer, objective, batch):
    optimizer.zero_grad()
    input_nodes, output_nodes, mfgs = batch
    inputs = mfgs[0].srcdata["feat"]
    labels = mfgs[-1].dstdata["label"]
    predictions = model(mfgs, inputs)
    loss = objective(predictions, labels)
    loss.backward()
    optimizer.step()
    return labels, predictions.argmax(1), loss


def update_dp(model, optimizer, objective, batch, g, clip_grad, clip_node, ns, trim_rule, device):
    optimizer.zero_grad()
    dst_node, subgraphs = batch
    appear_dict = AppearDict(roots=dst_node, subgraphs=subgraphs, trimming_rule=trim_rule, k=clip_node)
    # appear_dict.print_nodes()
    appear_dict.trim()
    temp_par = {}
    loss_batch = 0
    train_targets = []
    train_outputs = []
    for p in model.named_parameters():
        temp_par[p[0]] = torch.zeros_like(p[1])
    for i, root in enumerate(dst_node.tolist()):
        for p in model.named_parameters():
            p[1].grad = torch.zeros_like(p[1])
        blocks = appear_dict.build_blocks(root=root, graph=g)
        inputs = blocks[0].srcdata['feat']
        labels = blocks[-1].dstdata['label']
        predictions = model(blocks, inputs)
        loss = objective(predictions, labels)
        loss_batch += loss.item()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad, norm_type=2)
        for p in model.named_parameters():
            temp_par[p[0]] = temp_par[p[0]] + deepcopy(p[1].grad)
        pred = predictions.cpu().detach().argmax(1).numpy()
        train_targets.extend(labels.cpu().detach().numpy().astype(int).tolist())
        train_outputs.extend(pred)

    for p in model.named_parameters():
        p[1].grad = deepcopy(temp_par[p[0]]) + torch.normal(mean=0, std=ns * clip_node * clip_grad, size=temp_par[p[0]].size()).to(
            device)
        p[1].grad = p[1].grad / len(dst_node)
    optimizer.step()
    return train_targets, train_outputs, loss_batch


def eval_clean(model, objective, batch):
    input_nodes, output_nodes, mfgs = batch
    inputs = mfgs[0].srcdata["feat"]
    labels = mfgs[-1].dstdata["label"]
    predictions = model(mfgs, inputs)
    loss = objective(predictions, labels)
    return labels, predictions.argmax(1), loss


def train_fn(dataloader, model, criterion, optimizer, device, scheduler):
    model.to(device)
    model.train()
    train_targets = []
    train_outputs = []
    train_loss = 0
    for bi, d in enumerate(dataloader):
        target, pred, loss = update_clean(model=model, optimizer=optimizer, objective=criterion, batch=d)
        if scheduler is not None:
            scheduler.step()
        pred = pred.cpu().detach().numpy()
        train_targets.extend(target.cpu().detach().numpy().astype(int).tolist())
        train_outputs.extend(pred)
        train_loss += loss.item()
    return train_loss, train_outputs, train_targets


def train_dp(dataloader, model, criterion, optimizer, device, scheduler, g, clip_grad, clip_node, ns, trim_rule):
    model.to(device)
    g.to(device)
    model.train()
    batch = next(iter(dataloader))
    target, pred, loss = update_dp(model=model, optimizer=optimizer, objective=criterion, batch=batch, g=g,
                                   clip_grad=clip_grad, clip_node=clip_node, ns=ns, trim_rule=trim_rule, device=device)
    if scheduler is not None:
        scheduler.step()

    return loss, target, pred


def eval_fn(data_loader, model, criterion, device):
    model.to(device)
    fin_targets = []
    fin_outputs = []
    loss_eval = 0
    model.eval()
    with torch.no_grad():
        for bi, d in enumerate(data_loader):
            target, pred, loss = eval_clean(model=model, objective=criterion, batch=d)
            loss_eval += loss.item()
            outputs = pred.cpu().detach().numpy()
            fin_targets.extend(target.cpu().detach().numpy().astype(int).tolist())
            fin_outputs.extend(outputs)
    return loss_eval, fin_outputs, fin_targets


def performace_eval(args, y_true, y_pred):
    if args.performance_metric == 'acc':
        return accuracy_score(y_true=y_true, y_pred=y_pred)
    elif args.performance_metric == 'f1':
        return f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    elif args.performance_metric == 'auc':
        return roc_auc_score(y_true=y_true, y_score=y_pred)
    elif args.performance_metric == 'pre':
        return precision_score(y_true=y_true, y_pred=y_pred)
