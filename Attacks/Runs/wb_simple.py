import sys
import torch
import torchmetrics
import numpy as np
from loguru import logger
from hashlib import md5
from rich import print as rprint
from rich.pretty import pretty_repr
from Utils.utils import timeit
from Models.models import WbAttacker, NN
from Models.init import init_model, init_optimizer
from Attacks.Utils.utils import save_dict, plot_PCA
from Attacks.Utils.data_utils import test_distribution_shift
from Attacks.Utils.dataset import Data, ShadowData, custom_collate
from Attacks.Utils.train_eval import train_wb_attack, eval_att_wb_step, retrain, get_grad, train_attack, eval_attack_step
from functools import partial
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")


def run(args, graph, model, device, history, name):

    train_g, val_g, test_g, shadow_graph, graph_org = graph
    model_hist, att_hist = history

    with timeit(logger=logger, task='init-target-model'):

        if args.exist_model == False:
            rprint(f"Model is {args.exist_model} to exist, need to retrain")
            model_name = f"{md5(name['model'].encode()).hexdigest()}.pt"
            model, model_hist = retrain(args=args, train_g=train_g, val_g=val_g, test_g=test_g, model=model, 
                                        device=device, history=model_hist, name=model_name[:-3])
            
            target_model_name = f"{md5(name['model'].encode()).hexdigest()}.pkl"
            target_model_path = args.res_path + target_model_name
            save_dict(path=target_model_path, dct=model_hist)
        
    with timeit(logger=logger, task='preparing-shadow-data'):
        
        criterion = torch.nn.CrossEntropyLoss(reduction='none').to(device)
        # get grad from shadow graph
        grad_pos_tr, norm_pos_tr = get_grad(graph=shadow_graph, model=model, criterion=criterion, device=device, 
                               mask='pos_mask_tr', fan_out=[1,1])
        grad_pos_te, norm_pos_te = get_grad(graph=shadow_graph, model=model, criterion=criterion, device=device, 
                               mask='pos_mask_te', fan_out=[1,1])
        grad_neg_tr, norm_neg_tr = get_grad(graph=shadow_graph, model=model, criterion=criterion, device=device, 
                               mask='neg_mask_tr', fan_out=[1,1])
        grad_neg_te, norm_neg_te = get_grad(graph=shadow_graph, model=model, criterion=criterion, device=device, 
                               mask='neg_mask_te', fan_out=[1,1])
        
        norm_pos_tr = np.array(norm_pos_tr)
        norm_pos_te = np.array(norm_pos_te)
        norm_neg_tr = np.array(norm_neg_tr)
        norm_neg_te = np.array(norm_neg_te)

        x_tr = np.concatenate((norm_pos_tr, norm_neg_tr), axis=0)
        x_te = np.concatenate((norm_pos_te, norm_neg_te), axis=0)
        y_tr = np.concatenate((np.ones(len(norm_pos_tr)), np.zeros(len(norm_neg_tr))), axis=0)
        y_te = np.concatenate((np.ones(len(norm_pos_te)), np.zeros(len(norm_neg_te))), axis=0)

        arr = np.arange(len(x_tr))
        np.random.shuffle(arr)
        x_tr = x_tr[arr].reshape(-1, 1)
        y_tr = y_tr[arr]

        arr = np.arange(len(x_te))
        np.random.shuffle(arr)
        x_te = y_te[x_te].reshape(-1, 1) 
        y_te = y_te[arr]

        lr = LogisticRegression()
        lr.fit(X=x_tr, y=y_tr)
        rprint("Score for logistic regression:", lr.score(X=x_te, y=y_te))

        
    #     plot_PCA(gtrpos=grad_pos_tr, gtrneg=grad_neg_tr, gtepos=grad_pos_te, gteneg=grad_neg_te)
        
    #     rprint(f"Grad pos tr avg norm: {grad_pos_tr.norm() / grad_pos_tr.size(dim=0)}, neg tr avg norm: {grad_neg_tr.norm() / grad_neg_tr.size(dim=0)}")
    #     rprint(f"Grad pos te avg norm: {grad_pos_te.norm() / grad_pos_te.size(dim=0)}, neg te avg norm: {grad_neg_te.norm() / grad_neg_te.size(dim=0)}")

    #     norm_pos_tr = np.array(norm_pos_tr)
    #     norm_pos_te = np.array(norm_pos_te)
    #     norm_neg_tr = np.array(norm_neg_tr)
    #     norm_neg_te = np.array(norm_neg_te)
        
    #     rprint(f"Grad pos tr avg norm: {np.mean(norm_pos_tr)}, std {np.std(norm_pos_tr)}") 
    #     rprint(f"Grad neg tr avg norm: {np.mean(norm_neg_tr)}, std {np.std(norm_neg_tr)}") 
               

    #     rprint(f"Grad pos te avg norm: {np.mean(norm_pos_te)}, std {np.std(norm_pos_te)}") 
    #     rprint(f"Grad neg te avg norm: {np.mean(norm_neg_te)}, std {np.std(norm_neg_te)}") 

    #     x_tr = torch.cat((grad_pos_tr, grad_neg_tr), dim=0)
    #     y_tr = torch.cat((torch.ones(grad_pos_tr.size(dim=0)), torch.zeros(grad_neg_tr.size(dim=0))), dim=0)

    #     x_te = torch.cat((grad_pos_te, grad_neg_te), dim=0)
    #     y_te = torch.cat((torch.ones(grad_pos_te.size(dim=0)), torch.zeros(grad_neg_te.size(dim=0))), dim=0)

    #     id_xtr = range(x_tr.size(dim=0))
    #     id_ytr = y_tr.tolist()

    #     id_tr, id_val, y_tr_id, y_va_id = train_test_split(id_xtr, id_ytr, test_size=0.2, stratify=id_ytr)
        
    #     x_va = x_tr[id_val]
    #     y_va = y_tr[id_val]
    #     perm = torch.randperm(x_va.size(dim=0)).to(device)
    #     x_va = x_va[perm]
    #     y_va = y_va[perm]

    #     x_tr = x_tr[id_tr]
    #     y_tr = y_tr[id_tr]
    #     perm = torch.randperm(x_tr.size(dim=0)).to(device)
    #     x_tr = x_tr[perm]
    #     y_tr = y_tr[perm]

    #     test_distribution_shift(x_tr=x_tr, x_te=x_va)

    #     perm = torch.randperm(x_te.size(dim=0)).to(device)
    #     x_te = x_te[perm]
    #     y_te = y_te[perm]

    #     shtr_dataset = Data(X=x_tr, y=y_tr)
    #     shva_dataset = Data(X=x_va, y=y_va)
    #     shte_dataset = Data(X=x_te, y=y_te)
        

    # # # device = torch.device('cpu')
    # with timeit(logger=logger, task='train-attack-model'):
        
    #     tr_loader = torch.utils.data.DataLoader(shtr_dataset, batch_size=args.att_bs,
    #                                             drop_last=True, shuffle=True)

    #     va_loader = torch.utils.data.DataLoader(shva_dataset, batch_size=args.att_bs,
    #                                             shuffle=False, drop_last=False)

    #     te_loader = torch.utils.data.DataLoader(shte_dataset, batch_size=args.att_bs,
    #                                             shuffle=False, drop_last=False)

    #     att_model = NN(input_dim=x_tr.size(dim=1), hidden_dim=args.att_hid_dim, output_dim=1, n_layer=args.att_layers, dropout=0.2)
    #     att_model.to(device)
    #     att_opt = init_optimizer(optimizer_name=args.optimizer, model=att_model, lr=args.att_lr)
    #     att_model = train_attack(args=args, tr_loader=tr_loader, va_loader=va_loader, te_loader=te_loader,
    #                              attack_model=att_model, epochs=args.att_epochs, optimizer=att_opt, name=name['att'],
    #                              device=device, history=att_hist)

    # att_model.load_state_dict(torch.load(args.save_path + f"{name['att']}_attack.pt"))
    # metric = ['auc', 'acc', 'pre', 'rec', 'f1']
    # metric_dict = {
    #     'auc': torchmetrics.classification.BinaryAUROC().to(device),
    #     'acc': torchmetrics.classification.BinaryAccuracy().to(device),
    #     'pre': torchmetrics.classification.BinaryPrecision().to(device),
    #     'rec': torchmetrics.classification.BinaryRecall().to(device),
    #     'f1': torchmetrics.classification.BinaryF1Score().to(device)
    # }
    # for met in metric:
    #     te_loss, te_auc = eval_attack_step(model=att_model, device=device, loader=te_loader, 
    #                                        metrics=metric_dict[met], criterion=torch.nn.BCEWithLogitsLoss().to(device))
    #     rprint(f"Attack {met}: {te_auc}")
    
    return model_hist, att_hist


# pca = PCA()
# pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])

# # PCA on train

# X = torch.cat((grad_pos, grad_neg), dim=0).detach().cpu().numpy()
# y = torch.cat((torch.ones(grad_pos.size(dim=0)), torch.zeros(grad_neg.size(dim=0))), dim=0).numpy()

# pipe.fit(X)

# X_pos_tr = grad_pos_tr.detach().cpu().numpy()
# Xt = pipe.transform(X_pos_tr)
# plt.figure()
# plot = plt.scatter(Xt[:,0], Xt[:,1])
# plt.savefig('grad_pos_tr.jpg', bbox_inches='tight')

# X_neg_tr = grad_neg_tr.detach().cpu().numpy()
# Xt = pipe.transform(X_neg_tr)
# plt.figure()
# plot = plt.scatter(Xt[:,0], Xt[:,1])
# plt.savefig('grad_neg_tr.jpg', bbox_inches='tight')

# X_tr = torch.cat((grad_pos_tr, grad_neg_tr), dim = 0).detach().cpu().numpy()
# y_tr = torch.cat((torch.ones(grad_pos_tr.size(dim=0)), torch.zeros(grad_neg_tr.size(dim=0))), dim=0).numpy()
# Xt = pipe.transform(X_tr)
# plt.figure()
# plot = plt.scatter(Xt[:,0], Xt[:,1], c=y_tr)
# plt.legend(handles=plot.legend_elements()[0], labels=['pos', 'neg'])
# plt.savefig('grad_in_tr.jpg', bbox_inches='tight')

# X_pos_te = grad_pos_te.detach().cpu().numpy()
# Xt = pipe.transform(X_pos_te)
# plt.figure()
# plot = plt.scatter(Xt[:,0], Xt[:,1])
# plt.savefig('grad_pos_te.jpg', bbox_inches='tight')

# X_neg_te = grad_neg_te.detach().cpu().numpy()
# Xt = pipe.transform(X_neg_te)
# plt.figure()
# plot = plt.scatter(Xt[:,0], Xt[:,1])
# plt.savefig('grad_neg_te.jpg', bbox_inches='tight')

# X_te = torch.cat((grad_pos_te, grad_neg_te), dim = 0).detach().cpu().numpy()
# y_te = torch.cat((torch.ones(grad_pos_te.size(dim=0)), torch.zeros(grad_neg_te.size(dim=0))), dim=0).numpy()
# Xt = pipe.transform(X_te)
# plt.figure()
# plot = plt.scatter(Xt[:,0], Xt[:,1], c=y_te)
# plt.legend(handles=plot.legend_elements()[0], labels=['pos', 'neg'])
# plt.savefig('grad_in_te.jpg', bbox_inches='tight')