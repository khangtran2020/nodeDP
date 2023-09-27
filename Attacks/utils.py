import random
import os
import numpy as np
import torch
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


def get_model_name(history, mode='target', state='trained'):
    model_name = ''
    for k in history['general_keys']:
        model_name += f"{k}_{history['general'][k]}_"

    tar_model_key = deepcopy(history['target_model_keys'])
    
    if state == 'init':
        tar_model_key.remove('epochs')
        tar_model_key.remove('lr')

    if mode == 'target':
        if 'dp' in history.keys():
            for k in tar_model_key:
                model_name += f"{k}_{history['target_model'][k]}_"
            for k in history['dp_keys']:
                model_name += f"{k}_{history['dp'][k]}_"
        else:
            model_name = ''
            for k in tar_model_key:
                model_name += f"{k}_{history['target_model'][k]}_"
    else:
        for k in history['attack_model_keys']:
            model_name += f"{k}_{history['attack_model'][k]}_"
    model_name = model_name + state
    hashed = sha256(model_name.encode()).hexdigest()
    return f'{hashed}.pt'


def get_data_name(history):
    data_name = ''
    for k in history['general_keys']:
        data_name += f"{k}_{history['general'][k]}_"
    hashed = sha256(data_name.encode()).hexdigest()
    return f'{hashed}.pkl'

def get_his_name(args, current_date):
    dataset_str = f'{args.dataset}_run_{args.seed}_'
    date_str = f'{current_date.day}-{current_date.month}-{current_date.year}_{current_date.hour}-{current_date.minute}'
    if args.mode != 'mlp':
        model_str = f'{args.model_type}_{args.mode}_{args.epochs}_hops_{args.n_layers}_'
    else:
        model_str = f'{args.model_type}_{args.mode}_{args.mlp_mode}_{args.epochs}_hops_{args.n_layers}_'
    dp_str = f'{args.trim_rule}_M_{args.clip_node}_C_{args.clip}_sigma_{args.ns}_'
    desity_str = f'{args.submode}_{args.density}_'
    if args.mode == 'clean':
        if args.submode not in ['density', 'spectral', 'line', 'complete', 'tree']:
            res_str = dataset_str + model_str + date_str
        else:
            res_str = dataset_str + model_str + desity_str + date_str
    else:
        if args.submode not in ['density', 'spectral', 'line', 'complete', 'tree']:
            res_str = dataset_str + model_str + dp_str + date_str
        else:
            res_str = dataset_str + model_str + dp_str + desity_str + date_str
    return res_str


def init_history(args):

    general_keys = ['mode', 'seed']
    target_model_keys = ['model_type', 'n_layers', 'hid_dim', 'epochs', 'lr']
    dp_keys = ['ns', 'clip', 'clip_node', 'trim_rule', 'sampling_rate']
    attack_model_keys = ['attack_mode', 'attack_model_type', 'attack_n_layers', 'attack_hid_dim']

    general_setting = dict(zip(general_keys,[None for i in range(len(general_keys))]))
    target_model_setting = dict(zip(target_model_keys,[None for i in range(len(target_model_keys))]))
    dp_setting = dict(zip(dp_keys, [None for i in range(len(dp_keys))]))
    attack_model_setting = dict(zip(attack_model_keys, [None for i in range(len(attack_model_keys))]))

    for key in general_keys: general_setting[key] = getattr(args, key)
    for key in target_model_keys: target_model_setting[key] = getattr(args, key)
    for key in attack_model_keys: attack_model_setting[key] = getattr(args, key)
    if args.mode == 'nodedp':
        for key in dp_setting: dp_setting[key] = getattr(args, key)

    history = {
        'tr_id': None,
        'va_id': None,
        'te_id': None,
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
        'att_train_history_loss': [],
        'att_train_history_acc': [],
        'att_val_history_loss': [],
        'att_val_history_acc': [],
        'att_test_history_loss': [],
        'att_test_history_acc': [],
        'general_keys': general_keys,
        'target_model_keys': target_model_keys,
        'attack_model_keys': attack_model_keys,
        'general': general_setting,
        'target_model': target_model_setting,
        'attack_model': attack_model_setting
    }

    if args.mode == 'nodedp':
        history['dp_keys'] = dp_keys
        history['dp'] = dp_setting

    return history

def save_res(name, args, dct):
    save_name = args.res_path + name
    with open('{}.pkl'.format(save_name), 'wb') as f:
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
    general_keys = ['mode', 'seed', 'performance_metric', 'dataset', 'batch_size', 'n_neighbor', 'lr', 
                    'epochs', 'debug', 'device', 'submode']
    target_model_keys = ['model_type', 'n_layers', 'hid_dim']
    dp_keys = ['ns', 'clip', 'clip_node', 'trim_rule', 'sampling_rate']
    attack_model_keys = ['attack_mode', 'attack_model_type', 'attack_n_layers', 'attack_hid_dim']

    for key in general_keys: arg_dict[key] = getattr(args, key)
    for key in target_model_keys: arg_dict[key] = getattr(args, key)
    for key in attack_model_keys: arg_dict[key] = getattr(args, key)
    if args.mode == 'nodedp':
        for key in dp_keys: arg_dict[key] = getattr(args, key)
    rprint("Running experiments with hyper-parameters as follows: \n", pretty_repr(arg_dict))

def read_pickel(file):

    with open(file, 'rb') as f:
        res = pickle.load(f)
    return res