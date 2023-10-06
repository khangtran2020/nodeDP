import torch
import torchmetrics
from tqdm import tqdm
from dgl.dataloading import transforms
from dgl.dataloading.base import NID
from Models.train_eval import EarlyStopping
from rich import print as rprint
from Data.read import init_loader
from Models.init import init_optimizer
from Runs.run_clean import run as run_clean
from Runs.run_nodedp import run as run_nodedp

def get_entropy(pred):
    pass

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
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean')
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
        
        rprint(f"At epoch {epoch}: tr_loss {tr_loss}, tr_acc {tr_acc}, va_loss {va_loss}, va_acc {va_acc}")

        history['attr_loss'].append(tr_loss)
        history['attr_perf'].append(tr_acc)
        history['atva_loss'].append(va_loss)
        history['atva_perf'].append(va_acc)
        history['atte_loss'].append(te_loss)
        history['atte_perf'].append(te_acc)
        
        tk0.set_postfix(Loss=tr_loss, ACC=tr_acc.item(), Va_Loss=va_loss, Va_ACC=va_acc.item(), Te_ACC=te_acc.item())

        es(epoch=epoch, epoch_score=te_acc.item(), model=attack_model, model_path=args.save_path + model_name)

    return attack_model

def train_wb_attack(args, tr_loader, te_loader, weight, attack_model, epochs, optimizer, name, device, history):
    
    model_name = '{}_attack.pt'.format(name)

    attack_model.to(device)
    # DEfining criterion
    criterion = torch.nn.BCEWithLogitsLoss(reduction='mean', pos_weight=weight[0]/weight[1])
    criterion.to(device)

    metrics = torchmetrics.classification.BinaryAUROC().to(device)

    # DEfining Early Stopping Object
    es = EarlyStopping(patience=args.patience, verbose=False)

    # THE ENGINE LOOP
    tk0 = tqdm(range(epochs), total=epochs)
    for epoch in tk0:
        tr_loss, tr_acc = upd_att_wb_step(model=attack_model, device=device, loader=tr_loader, metrics=metrics,
                                             criterion=criterion, optimizer=optimizer)
        te_loss, te_acc = eval_att_wb_step(model=attack_model, device=device, loader=te_loader, metrics=metrics,
                                           criterion=criterion)
        
        rprint(f"At epoch {epoch}: tr_loss {tr_loss}, tr_acc {tr_acc.item()}, te_loss {te_loss}, te_acc {te_acc.item()}")

        history['attr_loss'].append(tr_loss)
        history['attr_perf'].append(tr_acc)
        history['atte_loss'].append(te_loss)
        history['atte_perf'].append(te_acc)
        
        tk0.set_postfix(Loss=tr_loss, ACC=tr_acc.item(), Te_Loss=te_loss, Te_ACC=te_acc.item())

        es(epoch=epoch, epoch_score=te_acc.item(), model=attack_model, model_path=args.save_path + model_name)

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
        predictions = torch.nn.functional.sigmoid(model(features))
        predictions = torch.squeeze(predictions, dim=-1)
        loss = criterion(predictions, target.float())
        loss.backward()
        optimizer.step()
        metrics.update(predictions, target)
        num_data += predictions.size(dim=0)
        train_loss += loss.item()*predictions.size(dim=0)
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
            predictions = torch.nn.functional.sigmoid(model(features))
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

    if args.mode == 'clean':
        run_mode = run_clean
    else:
        run_mode = run_nodedp

    model, history = run_mode(args=args, tr_info=tr_info, va_info=va_info, te_info=te_info, model=model,
                                      optimizer=optimizer, name=name, device=device, history=history)
    return model, history

def upd_att_wb_step(model, device, loader, metrics, criterion, optimizer):
    model.to(device)
    model.train()
    model.zero_grad()
    train_loss = 0
    num_data = 0.0
    for bi, d in enumerate(loader):
        optimizer.zero_grad()
        features, target = d
        target = torch.unsqueeze(target, dim=1).to(device)
        predictions = model(features)
        loss = criterion(predictions, target.float())
        loss.backward()
        optimizer.step()
        metrics.update(torch.nn.functional.sigmoid(predictions), target)
        num_data += predictions.size(dim=0)
        train_loss += loss.item()*predictions.size(dim=0)
    performance = metrics.compute()
    metrics.reset()
    return train_loss / num_data, performance

def eval_att_wb_step(model, device, loader, metrics, criterion):
    model.to(device)
    model.eval()
    val_loss = 0
    num_data = 0.0
    for bi, d in enumerate(loader):
        features, target = d
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

def get_grad(shadow_graph, target_graph, model, criterion, device, mask, pos=False):

    model.zero_grad()
    shadow_graph = shadow_graph.to(device)
    model.to(device)
    mask = shadow_graph.ndata[mask].int()

    pred_shadow = model.full(g=shadow_graph, x=shadow_graph.ndata['feat'])
    label_shadow = shadow_graph.ndata['label']
    loss_sh = criterion(pred_shadow, label_shadow)
    
    if pos:
        cos = []
        diff_norm = []
        norm_diff = []
        target_graph = target_graph.to(device)
        pred_target = model.full(g=target_graph, x=target_graph.ndata['feat'])
        label_target = target_graph.ndata['label']
        loss_tr = criterion(pred_target, label_target)

    grad_overall = torch.Tensor([]).to(device)
    norm = []

    for i, los in enumerate(loss_sh):

        if mask[i].item() > 0:
        
            los.backward(retain_graph=True)
            grad_sh = torch.Tensor([]).to(device)

            for name, p in model.named_parameters():
                if p.grad is not None:
                    new_grad = p.grad.detach().clone()
                    grad_sh = torch.cat((grad_sh, new_grad.flatten()), dim=0)
            model.zero_grad()

            if pos:
                
                id_tr = shadow_graph.ndata['id_intr'][i].item()
                grad_tr = torch.Tensor([]).to(device)
                loss_tr[id_tr].backward(retain_graph=True)
                for name, p in model.named_parameters():
                    if p.grad is not None:
                        new_grad = p.grad.detach().clone()
                        grad_tr = torch.cat((grad_tr, new_grad.flatten()), dim=0)
                model.zero_grad()
            
                c = (grad_sh*grad_tr).sum().item() / (grad_sh.norm(p=2).item() + grad_tr.norm(p=2).item() + 1e-12)
                n1 = (grad_sh.norm() - grad_tr.norm()).abs().item()
                n2 = (grad_sh - grad_tr).norm().item()
                cos.append(c)
                diff_norm.append(n1)
                norm_diff.append(n2)


            grad_sh = torch.unsqueeze(grad_sh, dim=0)
            norm.append(grad_sh.norm().detach().item())
            grad_overall = torch.cat((grad_overall, grad_sh), dim=0)

    if pos:

        rprint(f"For {mask}: average cosine {sum(cos) / (len(cos) + 1e-12)}, average diff in norm {sum(diff_norm) / (len(diff_norm) + 1e-12)}, average norm of diff {sum(norm_diff) / (len(norm_diff) + 1e-12)}")

    return grad_overall, norm

def sample_blocks(nodes, graph, n_layer, device, fout):
    blocks = []
    seed_nodes = nodes
    for i in reversed(range(n_layer)):
        frontier = graph.sample_neighbors(seed_nodes, fout[i], output_device=device)
        block = transforms.to_block(frontier, seed_nodes, include_dst_in_src=True)
        seed_nodes = block.srcdata[NID]
        blocks.insert(0, block)
    return blocks