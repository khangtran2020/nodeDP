import dgl
import torch
import torchmetrics

from tqdm import tqdm
from Models.train_eval import EarlyStopping, train_fn, eval_fn, performace_eval
from Utils.utils import get_name, save_res
from dgl.dataloading import NeighborSampler


def train_shadow(args, tr_loader, va_loader, shadow_model, epochs, optimizer, name, device, mode="hops"):
    
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
                                      criterion=criterion, optimizer=optimizer, mode=mode)
        va_loss, va_acc = eval_step(model=shadow_model, device=device, loader=va_loader, metrics=metrics,
                                    criterion=criterion, mode=mode)

        tk0.set_postfix(Loss=tr_loss, ACC=tr_acc.item(), Va_Loss=va_loss, Va_ACC=va_acc.item())

        es(epoch=epoch, epoch_score=va_acc.item(), model=shadow_model, model_path=args.save_path + model_name)

    return shadow_model

def update_step(model, device, loader, metrics, criterion, optimizer, mode):
    
    model.to(device)
    model.train()
    train_loss = 0
    num_data = 0.0

    for bi, d in enumerate(loader):
        optimizer.zero_grad()
        input_nodes, output_nodes, mfgs = d
        inputs = mfgs[0].srcdata["feat"]
        if mode == 'hops':
            labels = mfgs[-1].dstdata["tar_conf"]
        else:
            labels = mfgs[-1].dstdata["tar_conf_nohop"]
        predictions = model(mfgs, inputs)
        loss = criterion(predictions, labels)
        loss.backward()
        optimizer.step()
        metrics.update(predictions.argmax(dim=1), labels.argmax(dim=1))
        num_data += predictions.size(dim=0)
        train_loss += loss.item()*predictions.size(dim=0)

    performance = metrics.compute()
    metrics.reset()

    return train_loss / num_data, performance

def eval_step(model, device, loader, metrics, criterion, mode):
    model.to(device)
    model.eval()
    val_loss = 0
    num_data = 0.0
    with torch.no_grad():
        for bi, d in enumerate(loader):
            input_nodes, output_nodes, mfgs = d
            inputs = mfgs[0].srcdata["feat"]
            if mode == 'hops':
                labels = mfgs[-1].dstdata["tar_conf"]
            else:
                labels = mfgs[-1].dstdata["tar_conf_nohop"]
            predictions = model(mfgs, inputs)
            loss = criterion(predictions, labels)
            metrics.update(predictions.argmax(dim=1), labels.argmax(dim=1))
            num_data += predictions.size(dim=0)
            val_loss += loss.item()*predictions.size(dim=0)
        performance = metrics.compute()
        metrics.reset()
    return val_loss / num_data, performance

def train_attack(args, tr_loader, va_loader, te_loader, attack_model, epochs, optimizer, name, device):

    attack_model.to(device)
    model_name = name

    # DEfining criterion
    criterion = torch.nn.BCELoss(reduction='mean')
    criterion.to(device)

    metrics = torchmetrics.classification.BinaryAccuracy().to(device)

    # DEfining Early Stopping Object
    es = EarlyStopping(patience=args.patience, verbose=False)

    # THE ENGINE LOOP
    tk0 = tqdm(range(epochs), total=epochs)
    for epoch in tk0:
        tr_loss, tr_acc = update_attack_step(model=attack_model, device=device, loader=tr_loader, metrics=metrics,
                                             criterion=criterion, optimizer=optimizer)
        va_loss, va_acc, va_topk = eval_attack_step(model=attack_model, device=device, loader=va_loader, metrics=metrics,
                                        criterion=criterion, rate=args.topk_rate)
        te_loss, te_acc, te_topk = eval_attack_step(model=attack_model, device=device, loader=te_loader, metrics=metrics,
                                           criterion=criterion, rate=args.topk_rate)
        
        tk0.set_postfix(Loss=tr_loss, ACC=tr_acc.item(), Va_Loss=va_loss, Va_ACC=va_acc.item(), VA_TOPK=va_topk.item(), Te_ACC=te_acc.item(), Te_TOPK=te_topk.item())

        es(epoch=epoch, epoch_score=va_acc.item(), model=attack_model, model_path=args.save_path + model_name)

    return attack_model

def update_attack_step(model, device, loader, metrics, criterion, optimizer):
    model.to(device)
    model.train()
    model.zero_grad()
    train_loss = 0
    num_data = 0.0
    for bi, d in enumerate(loader):
        optimizer.zero_grad()
        features, target = d
        features = features.to(device)
        target = target.to(device)
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

def eval_attack_step(model, device, loader, metrics, criterion, rate):
    model.to(device)
    model.eval()
    val_loss = 0
    num_data = 0.0
    pred = torch.Tensor([]).to(device)
    label = torch.Tensor([]).to(device)
    entr = torch.Tensor([]).to(device)
    with torch.no_grad():
        for bi, d in enumerate(loader):
            features, target = d
            features = features.to(device)
            target = target.to(device)
            predictions = model(features)
            predictions = torch.squeeze(predictions, dim=-1)
            entropy = get_binary_entropy(pred=predictions)
            pred = torch.cat((pred, predictions), dim = 0)
            label = torch.cat((label, target), dim=0)
            entr = torch.cat((entr, entropy), dim=0)
            loss = criterion(predictions, target.float())
            metrics.update(predictions, target)
            num_data += predictions.size(dim=0)
            val_loss += loss.item()*predictions.size(dim=0)
        performance = metrics.compute()
        metrics.reset()
    val, indx = torch.topk(entr, int(rate*entr.size(dim=0)), largest=False)
    pred_new = pred[indx]
    label_new = label[indx]
    performance_topk = metrics(pred_new, label_new)
    return val_loss/num_data, performance, performance_topk

def get_entropy(pred):
    log_pred = torch.log2(pred+1e-12)
    temp = -1*pred*log_pred
    return temp.sum(dim=-1)

def get_binary_entropy(pred):
    log_pred = torch.log2(pred+1e-12)
    temp = -1*pred*log_pred
    return temp