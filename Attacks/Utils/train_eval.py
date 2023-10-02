import torch
import torchmetrics
from tqdm import tqdm
from Models.train_eval import EarlyStopping
from rich import print as rprint
from Data.read import init_loader
from Models.init import init_optimizer
from Runs.run_clean import run as run_clean
from Runs.run_nodedp import run as run_nodedp


def train_shadow(args, tr_loader, shadow_model, epochs, optimizer, name, device, history, mode):
    model_name = f'{name}_{mode}_shadow.pt'
    model_path = args.save_path + model_name
    shadow_model.to(device)

    # DEfining criterion
    criterion = torch.nn.CrossEntropyLoss()
    criterion.to(device)

    metrics = torchmetrics.classification.Accuracy(task="multiclass", num_classes=args.num_class).to(device)

    # THE ENGINE LOOP
    tk0 = tqdm(range(epochs), total=epochs)
    for epoch in tk0:
        tr_loss, tr_acc = update_step(model=shadow_model, device=device, loader=tr_loader, 
                                      metrics=metrics, criterion=criterion, optimizer=optimizer, mode=mode)
        history['shtr_loss'].append(tr_loss)
        history['shtr_perf'].append(tr_acc)
        torch.save(shadow_model.state_dict(), model_path)
        tk0.set_postfix(Loss=tr_loss, ACC=tr_acc.item())

    return shadow_model

def update_step(model, device, loader, metrics, criterion, optimizer, mode):
    model.to(device)
    model.train()
    train_loss = 0
    num_data = 0.0
    conf = 'tar_conf' if mode == 'hops' else 'tar_conf_nohop'
    for bi, d in enumerate(loader):
        optimizer.zero_grad()
        input_nodes, output_nodes, mfgs = d
        inputs = mfgs[0].srcdata["feat"]
        labels = mfgs[-1].dstdata[conf]
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

def eval_step(model, device, loader, metrics, criterion):
    model.to(device)
    model.eval()
    val_loss = 0
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
            val_loss += loss.item()*predictions.size(dim=0)
        performance = metrics.compute()
        metrics.reset()
    return val_loss / num_data, performance

def train_attack(args, tr_loader, va_loader, te_loader, attack_model, epochs, optimizer, name, device, history):
    
    model_name = '{}_attack.pt'.format(name)

    attack_model.to(device)

    # DEfining criterion
    criterion = torch.nn.BCELoss(reduction='mean')
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
        
        history['attr_loss'].append(tr_loss)
        history['attr_perf'].append(tr_acc)
        history['atva_loss'].append(va_loss)
        history['atva_perf'].append(va_acc)
        history['atte_loss'].append(te_loss)
        history['atte_perf'].append(te_acc)
        
        tk0.set_postfix(Loss=tr_loss, ACC=tr_acc.item(), Va_Loss=va_loss, Va_ACC=va_acc.item(), Te_ACC=te_acc.item())

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

def eval_attack_step(model, device, loader, metrics, criterion):
    model.to(device)
    model.eval()
    val_loss = 0
    num_data = 0.0
    with torch.no_grad():
        for bi, d in enumerate(loader):
            features, target = d
            features = features.to(device)
            target = target.to(device)
            predictions = model(features)
            predictions = torch.squeeze(predictions, dim=-1)
            loss = criterion(predictions, target.float())
            metrics.update(predictions, target)
            num_data += predictions.size(dim=0)
            val_loss += loss.item()*predictions.size(dim=0)
        performance = metrics.compute()
        metrics.reset()
    return val_loss/num_data, performance

def retrain(args, train_g, val_g, test_g, model, device, history, name):

    train_g = train_g.to(device)
    val_g = val_g.to(device)
    test_g = test_g.to(device)
    rprint(f"Node feature device: {train_g.ndata['feat'].device}, Node label device: {train_g.ndata['feat'].device}, "
           f"Src edges device: {train_g.edges()[0].device}, Dst edges device: {train_g.edges()[1].device}")
    
    tr_loader, va_loader, te_loader = init_loader(args=args, device=device, train_g=train_g, test_g=test_g, val_g=val_g)
    optimizer = init_optimizer(optimizer_name=args.optimizer, model=model, lr=args.lr)

    tr_info = (train_g, tr_loader)
    va_info = va_loader
    te_info = (test_g, te_loader)

    if args.tar_clean == 1:
        run_mode = run_clean
    else:
        run_mode = run_nodedp

    model, history = run_mode(args=args, tr_info=tr_info, va_info=va_info, te_info=te_info, model=model,
                                      optimizer=optimizer, name=name, device=device, history=history)
    return model, history