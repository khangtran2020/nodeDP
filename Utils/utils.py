import random
import os
import numpy as np
import torch
import time
import pickle
from contextlib import contextmanager
from rich import print as rprint
from rich.pretty import pretty_repr

# from rich.tree import


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


def get_name(args, current_date):
    dataset_str = f'{args.dataset}_run_{args.seed}_'
    date_str = f'{current_date.day}-{current_date.month}-{current_date.year}_{current_date.hour}-{current_date.minute}'
    model_str = f'{args.mode}_{args.epochs}_hops_{args.n_layers}_'
    dp_str = f'{args.trim_rule}_M_{args.clip_node}_C_{args.clip}_sigma_{args.ns}_'
    desity_str = f'density_{args.density}_'
    if args.mode == 'clean':
        if args.submode != 'density':
            res_str = dataset_str + model_str + date_str
        else:
            res_str = dataset_str + model_str + desity_str + date_str
    else:
        if args.submode != 'density':
            res_str = dataset_str + model_str + dp_str + date_str
        else:
            res_str = dataset_str + model_str + dp_str + desity_str + date_str
    return res_str


def save_res(name, args, dct):
    save_name = args.res_path + name
    with open('{}.pkl'.format(save_name), 'wb') as f:
        pickle.dump(dct, f)


def get_index_by_value(a, val):
    return (a == val).nonzero(as_tuple=True)[0]

def get_index_by_list(arr, test_arr):
     return torch.isin(arr, test_arr).nonzero(as_tuple=True)[0]

def get_index_bynot_value(a, val):
    return (a != val).nonzero(as_tuple=True)[0]


def print_args(args):
    arg_dict = {}
    keys = ['mode', 'seed', 'performance_metric', 'dataset', 'batch_size', 'n_neighbor', 'model_type', 'lr', 'n_layers',
            'epochs', 'clip', 'clip_node', 'trim_rule', 'ns', 'debug']
    for key in keys:
        arg_dict[key] = getattr(args, key)

    rprint("Running experiments with hyper-parameters as follows: \n", pretty_repr(arg_dict))
    # print(getattr(args, )

def print_dict(dict_, name):
    rprint(f"Dictionary of {name}: \n", pretty_repr(dict_))
