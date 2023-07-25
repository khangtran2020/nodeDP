import dgl
import torch
import torchmetrics

from tqdm import tqdm
from Models.train_eval import EarlyStopping, train_fn, eval_fn, performace_eval
from Utils.utils import get_name, save_res
from dgl.dataloading import NeighborSampler


def train_shadow(args, tr_loader, va_loader, shadow_model, epochs, optimizer, name, device):
    model_name = '{}_shadow.pt'.format(name)

    shadow_model.to(device)

    # DEfining criterion
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)

    metrics = torchmetrics.classification.Accuracy(task="multiclass", num_classes=args.num_class).to(device)

    # DEfining Early Stopping Object
    es = EarlyStopping(patience=args.patience, verbose=False)

    # THE ENGINE LOOP
    tk0 = tqdm(range(epochs), total=epochs)
    for epoch in tk0:
        tr_loss, tr_acc = update_step(model=shadow_model, device=device, loader=tr_loader, metrics=metrics,
                                      criterion=criterion, optimizer=optimizer)
        va_loss, va_acc = eval_step(model=shadow_model, device=device, loader=va_loader, metrics=metrics,
                                    criterion=criterion)

        tk0.set_postfix(Loss=tr_loss, ACC=tr_acc.item(), Va_Loss=va_loss, Va_ACC=va_acc.item())

        es(epoch=epoch, epoch_score=va_acc.item(), model=shadow_model, model_path=args.save_path + model_name)

    return shadow_model


def update_step(model, device, loader, metrics, criterion, optimizer):
    model.to(device)
    model.train()
    train_loss = 0
    num_data = 0.0
    for bi, d in enumerate(loader):
        optimizer.zero_grad()
        input_nodes, output_nodes, mfgs = d
        inputs = mfgs[0].srcdata["feat"]
        labels = mfgs[-1].dstdata["tar_conf"]
        predictions = model(mfgs, inputs)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        metrics.update(predictions.argmax(dim=1), labels.argmax(dim=1))
        num_data += predictions.size(dim=0)
        train_loss += loss.item()
    performance = metrics.compute()
    metrics.reset()
    return train_loss / num_data, performance


def eval_step(model, device, loader, metrics, criterion):
    model.to(device)
    model.eval()
    loss = 0
    num_data = 0.0
    with torch.no_grad():
        for bi, d in enumerate(loader):
            input_nodes, output_nodes, mfgs = d
            inputs = mfgs[0].srcdata["feat"]
            labels = mfgs[-1].dstdata["tar_conf"]
            predictions = model(mfgs, inputs)
            loss = criterion(predictions, labels)
            metrics.update(predictions.argmax(dim=1), labels.argmax(dim=1))
            num_data += predictions.size(dim=0)
            loss += loss.item()
        performance = metrics.compute()
        metrics.reset()
    return loss / num_data, performance


def train_attack(args, tr_loader, va_loader, te_loader, attack_model, epochs, optimizer, name, device):
    model_name = '{}_attack.pt'.format(name)

    attack_model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss()
    criterion.to(device)

    metrics = torchmetrics.classification.BinaryAUROC().to(device)

    # DEfining Early Stopping Object
    es = EarlyStopping(patience=args.patience, verbose=False)

    # THE ENGINE LOOP
    tk0 = tqdm(range(epochs), total=epochs)
    for epoch in tk0:
        tr_loss, tr_acc = update_attack_step(model=attack_model, device=device, loader=tr_loader, metrics=metrics,
                                             criterion=criterion, optimizer=optimizer)
        va_loss, va_acc = eval_attack_step(model=attack_model, device=device, loader=va_loader, metrics=metrics,
                                           criterion=criterion)
        te_loss, te_acc = eval_attack_step(model=attack_model, device=device, loader=te_loader, metrics=metrics,
                                           criterion=criterion)

        tk0.set_postfix(Loss=tr_loss, ACC=tr_acc.item(), Va_Loss=va_loss, Va_ACC=va_acc.item(), Te_ACC=te_acc.item())

        es(epoch=epoch, epoch_score=va_acc.item(), model=attack_model, model_path=args.save_path + model_name)

    return attack_model


def update_attack_step(model, device, loader, metrics, criterion, optimizer):
    model.to(device)
    model.train()
    train_loss = 0
    num_data = 0.0
    for bi, d in enumerate(loader):
        optimizer.zero_grad()
        features, target = d
        predictions = model(features)
        predictions = torch.squeeze(predictions, dim=-1)
        loss = criterion(predictions, target.float())
        loss.backward()
        optimizer.step()
        metrics.update(predictions, target)
        num_data += predictions.size(dim=0)
        train_loss += loss.item()
    performance = metrics.compute()
    metrics.reset()
    return train_loss / num_data, performance


def eval_attack_step(model, device, loader, metrics, criterion):
    model.to(device)
    model.eval()
    loss = 0
    num_data = 0.0
    with torch.no_grad():
        for bi, d in enumerate(loader):
            features, target = d
            predictions = model(features)
            predictions = torch.squeeze(predictions, dim=-1)
            loss = criterion(predictions, target.float())
            metrics.update(predictions, target)
            num_data += predictions.size(dim=0)
            loss += loss.item()
        performance = metrics.compute()
        metrics.reset()
    return loss / num_data, performance
