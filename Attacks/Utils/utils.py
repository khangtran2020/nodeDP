import random
import os
import numpy as np
import torch
import dgl
import time
import pickle
from contextlib import contextmanager
from rich import print as rprint
from rich.pretty import pretty_repr
from hashlib import sha256
from copy import deepcopy

@contextmanager
def timeit(logger, task):
    logger.info(f'Started task {task} ...')
    t0 = time.time()
    yield
    t1 = time.time()
    logger.info(f'Completed task {task} - {(t1 - t0):.3f} sec.')

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def generate_name(args):

    key_dict = {
        'dataset': '',
        'mode': 'mode',
        'seed': 'seed',
        'n_neighbor': 'nnei',
        'model_type': '',
        'n_layers': 'hops',
        'hid_dim': 'hdim',
        'epochs': 'epch',
        'lr': 'lr',
        'ns': 'sigm',
        'clip': 'clip',
        'clip_node': 'M',
        'trim_rule': 'rule',
        'sampling_rate': 'q'
    }

    if args.mode == 'nodedp':
        keys = ['dataset', 'mode', 'seed', 'n_neighbor', 'model_type', 'n_layers', 'hid_dim',
                'epochs', 'lr', 'ns', 'clip', 'clip_node', 'trim_rule', 'sampling_rate']
    else:
        keys = ['dataset', 'mode', 'seed', 'n_neighbor', 'model_type', 'n_layers', 'hid_dim',
                'epochs', 'lr']

    res_str = ''
    for i, key in enumerate(keys):
        val = getattr(args, key)
        name = key_dict[key]
        if i < len(keys) - 1:
            if name != '':
                res_str += f'{val}_'
            else:
                res_str += f'{name}_{val}_'
        else:
            if name != '':
                res_str += f'{val}'
            else:
                res_str += f'{name}_{val}'

    hashed = sha256(res_str.encode()).hexdigest()

    names = {
        'tar_model_init': f"target_{hashed}_init.pt",
        'tar_model_trained': f"target_{hashed}_trained.pt",
        'history': f"{hashed}.pkl",
        'name': hashed,
        'human': res_str
    }

    return names

def init_history(args):

    history = {
        'exp_setting': {

        },

        'targeted_model': {
            'train_history_loss': [],
            'train_history_acc': [],
            'val_history_loss': [],
            'val_history_acc': [],
            'test_history_loss': [],
            'test_history_acc': [],
            '% subgraph': [],
            '% node avg': [],
            '% edge avg': [],
            'avg rank': [],
            'best_test': 0,
        },
        'attack_model': {
            'att_train_history_loss': [],
            'att_train_history_acc': [],
            'att_val_history_loss': [],
            'att_val_history_acc': [],
            'att_test_history_loss': [],
            'att_test_history_acc': [],
        }
    }


    if args.mode == 'nodedp':
        keys = ['dataset', 'mode', 'seed', 'n_neighbor', 'model_type', 'n_layers', 'hid_dim',
                'epochs', 'lr', 'ns', 'clip', 'clip_node', 'trim_rule', 'sampling_rate', 'batch_size']
    else:
        keys = ['dataset', 'mode', 'seed', 'n_neighbor', 'model_type', 'n_layers', 'hid_dim',
                'epochs', 'lr', 'batch_size']

    for key in keys: history['exp_setting'][key] = getattr(args, key)
    return history

def save_dict(path, dct):
    with open(path, 'wb') as f:
        pickle.dump(dct, f)

def get_index_by_value(a, val):
    return (a == val).nonzero(as_tuple=True)[0]

def get_index_by_list(arr, test_arr):
    return torch.isin(arr, test_arr).nonzero(as_tuple=True)[0]

def get_index_by_not_list(arr, test_arr):
    return (1 - torch.isin(arr, test_arr).int()).nonzero(as_tuple=True)[0]

def get_index_bynot_value(a, val):
    return (a != val).nonzero(as_tuple=True)[0]

def print_args(args):
    arg_dict = {}
    if args.mode == 'nodedp':
        keys = ['dataset', 'mode', 'seed', 'n_neighbor', 'model_type', 'n_layers', 'hid_dim',
                'epochs', 'lr', 'ns', 'clip', 'clip_node', 'trim_rule', 'sampling_rate', 'batch_size']
    else:
        keys = ['dataset', 'mode', 'seed', 'n_neighbor', 'model_type', 'n_layers', 'hid_dim',
                'epochs', 'lr', 'batch_size']

    for key in keys: arg_dict[key] = getattr(args, key)
    rprint("Running experiments with hyper-parameters as follows: \n", pretty_repr(arg_dict))

def read_pickel(file):
    with open(file, 'rb') as f:
        res = pickle.load(f)
    return res

def generate_nohop_graph(graph):

    nodes = graph.nodes()
    num_node = nodes.size(dim=0)

    g = dgl.graph((nodes.tolist(), nodes.tolist()), num_nodes=num_node)
    for key in graph.ndata.keys():
        g.ndata[key] = graph.ndata[key].clone()

    return g


