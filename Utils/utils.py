import random
import os
import numpy as np
import torch
import time
import pickle
from contextlib import contextmanager


@contextmanager
def timeit(logger, task):
    logger.info('Started task %s ...', task)
    t0 = time.time()
    yield
    t1 = time.time()
    logger.info('Completed task %s - %.3f sec.', task, t1-t0)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_name(args, current_date, fold=0):
    dataset_str = f'{args.dataset}_{fold}_{args.ratio}_'
    date_str = f'{current_date.day}-{current_date.month}-{current_date.year}_{current_date.hour}-{current_date.minute}-{current_date.second}'
    model_str = f'{args.mode}_{args.epochs}_{args.performance_metric}_{args.optimizer}_'
    dp_str = f'{args.trim_rule}_{args.clip_node}_{args.clip}_{args.ns}_'
    if args.mode == 'clean':
        res_str = dataset_str + model_str + date_str
    elif args.mode == 'dp':
        res_str = dataset_str + model_str + dp_str + date_str
    return res_str

def save_res(name, args, dct):
    save_name = args.res_path + name
    with open('{}.pkl'.format(save_name), 'wb') as f:
        pickle.dump(dct, f)


def get_index_by_value(a, val):
    return (a == val).nonzero(as_tuple=True)[0]

def get_index_bynot_value(a, val):
    return (a != val).nonzero(as_tuple=True)[0]