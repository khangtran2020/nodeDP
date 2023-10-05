import torch
import pickle
import matplotlib.pyplot as plt
from rich import print as rprint
from rich.pretty import pretty_repr
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

plt.rcParams["figure.figsize"] = (3.5, 3)
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize']= 12
plt.rcParams['legend.title_fontsize']= 14


def save_dict(path, dct):
    with open(path, 'wb') as f:
        pickle.dump(dct, f)

def print_args(args):
    arg_dict = {}
    if args.mode == 'dp':
        keys = ['mode', 'seed', 'performance_metric', 'dataset', 'batch_size', 'n_neighbor', 'model_type', 
                'lr', 'n_layers', 'hid_dim', 'epochs', 'clip', 'clip_node', 'trim_rule', 'ns', 'debug', 'device', 
                'sampling_rate', 'optimizer', 'att_mode', 'att_submode', 'att_layers', 'att_hid_dim', 'att_lr', 'att_bs',
                'att_epochs', 'sha_lr', 'sha_epochs', 'sha_ratio']
    else:
        keys = ['mode', 'seed', 'performance_metric', 'dataset', 'batch_size', 'n_neighbor', 'model_type', 
                'lr', 'n_layers', 'hid_dim', 'epochs', 'debug', 'device', 'optimizer', 'att_mode', 'att_submode', 
                'att_layers', 'att_hid_dim', 'att_lr', 'att_bs', 'att_epochs', 'sha_lr', 'sha_epochs', 'sha_ratio']
        
    for key in keys:
        arg_dict[key] = getattr(args, key)

    rprint("Running experiments with hyper-parameters as follows: \n", pretty_repr(arg_dict))

def init_history(args):

    data_hist = {
        'tr_id': None,
        'va_id': None,
        'te_id': None,
    }

    if args.mode == 'clean':

        target_model_hist = {
            'name': None,
            'train_history_loss': [],
            'train_history_acc': [],
            'val_history_loss': [],
            'val_history_acc': [],
            'test_history_loss': [],
            'test_history_acc': [],
            'best_test': 0
        }

    else:

        target_model_hist =  {
            'name': None,
            'train_history_loss': [],
            'train_history_acc': [],
            'val_history_loss': [],
            'val_history_acc': [],
            'test_history_loss': [],
            'test_history_acc': [],
            '% subgraph': [],
            '% node avg': [],
            '% edge avg': [],
            'avg rank': []
        }

    att_hist = {
        'str_mask': None,
        'ste_mask': None,
        'shtr_loss': [],
        'shtr_perf': [],
        'attr_loss': [],
        'attr_perf': [],
        'atva_loss': [],
        'atva_perf': [],
        'atte_loss': [],
        'atte_perf': [],
    }

    return data_hist, target_model_hist, att_hist

def get_name(args, current_date):

    date_str = f'{current_date.day}{current_date.month}{current_date.year}-{current_date.hour}{current_date.minute}'

    data_key = ['dataset', 'seed']
    model_key = ['dataset', 'mode', 'seed', 'n_neighbor', 'model_type', 'lr', 'n_layers', 'hid_dim', 'epochs', 'optimizer']
    dp_key = ['clip', 'clip_node', 'trim_rule', 'ns', 'sampling_rate']

    if args.mode == 'nodedp':
        gen_keys = ['dataset', 'mode', 'seed', 'n_neighbor', 'model_type', 'n_layers', 'clip', 'clip_node', 'trim_rule', 'ns',
                'sampling_rate', 'att_mode', 'sha_ratio']
    else:
        gen_keys = ['dataset', 'mode', 'seed', 'n_neighbor', 'model_type', 'n_layers', 'att_mode', 'sha_ratio']

    if args.att_mode == 'blackbox':
        att_key = ['att_mode', 'att_submode', 'att_layers', 'att_hid_dim', 'att_lr', 'att_bs', 
                'att_epochs', 'sha_lr', 'sha_epochs', 'sha_ratio']
    else:
        att_key = ['att_mode', 'att_submode', 'att_layers', 'att_hid_dim', 'att_lr', 'att_bs',
                    'att_epochs', 'sha_ratio']
    
    general_str = ''
    for key in gen_keys:
        general_str += f"{key}_{getattr(args, key)}_"
    general_str += date_str

    data_str = ''
    for key in data_key:
        data_str += f"{key}_{getattr(args, key)}_"
    
    model_str = ''
    for key in model_key:
        model_str += f"{key}_{getattr(args, key)}_"

    dp_str = ''
    if args.mode == 'nodedp':
        for key in dp_key:
            dp_str += f"{key}_{getattr(args, key)}_"
    
    if args.mode == 'nodedp':
        model_str += dp_str

    att_str = ''
    for key in att_key:
        att_str += f"{key}_{getattr(args, key)}_"

    name = {
        'data': data_str[:-1],
        'model': model_str[:-1],
        'att': att_str[:-1],
        'general': general_str
    }

    return name

def plot_PCA(gtrpos, gtrneg, gtepos, gteneg):

    pca = PCA()
    pipe = Pipeline([('scaler', StandardScaler()), ('pca', pca)])
    fig, ax = plt.subplots(2, 3, sharex='col', sharey='row', figsize=(15, 8))  

    X = torch.cat((gtrpos, gtepos, gtrneg, gteneg), dim=0).detach().cpu().numpy()
    pipe.fit(X)

    X_tr = torch.cat((gtrpos, gtrneg), dim = 0).detach().cpu().numpy()
    y_tr = torch.cat((torch.ones(gtrpos.size(dim=0)), torch.zeros(gtrneg.size(dim=0))), dim=0).numpy()
    X_tr = pipe.transform(X_tr)

    plot = ax[0][0].scatter(X_tr[:,0], X_tr[:,1], c=y_tr)
    ax[0][0].set_title("Gradient\nIn shaddow train")
    plt.legend(handles=plot.legend_elements()[0], labels=['neg', 'pos'])

    plot = ax[0][1].scatter(X_tr[:,1], X_tr[:,2], c=y_tr)
    ax[0][1].set_title("Gradient\nIn shaddow train")
    plt.legend(handles=plot.legend_elements()[0], labels=['neg', 'pos'])

    plot = ax[0][2].scatter(X_tr[:,0], X_tr[:,2], c=y_tr)
    ax[0][2].set_title("Gradient\nIn shaddow train")
    plt.legend(handles=plot.legend_elements()[0], labels=['neg', 'pos'])

    X_te = torch.cat((gtepos, gteneg), dim = 0).detach().cpu().numpy()
    y_te = torch.cat((torch.ones(gtepos.size(dim=0)), torch.zeros(gteneg.size(dim=0))), dim=0).numpy()
    X_te = pipe.transform(X_te)
    
    plot = ax[1][0].scatter(X_te[:,0], X_te[:,1], c=y_te)
    ax[1][0].set_title("Gradient\nIn shaddow test")
    plt.legend(handles=plot.legend_elements()[0], labels=['neg', 'pos'])

    plot = ax[1][1].scatter(X_te[:,1], X_te[:,2], c=y_te)
    ax[1][1].set_title("Gradient\nIn shaddow test")
    plt.legend(handles=plot.legend_elements()[0], labels=['neg', 'pos'])

    plot = ax[1][2].scatter(X_te[:,0], X_te[:,2], c=y_te)
    ax[1][2].set_title("Gradient\nIn shaddow test")
    plt.legend(handles=plot.legend_elements()[0], labels=['neg', 'pos'])

    plt.savefig('results/grad_inspect.jpg', bbox_inches='tight')





