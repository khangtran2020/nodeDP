import random
import os
import numpy as np
import torch
import time
import pickle
from contextlib import contextmanager
from rich import print as rprint
from rich.pretty import pretty_repr

def print_args(args):
    arg_dict = {}
    if args.mode == 'dp':
        keys = ['mode', 'seed', 'performance_metric', 'dataset', 'batch_size', 'n_neighbor', 'model_type', 
                'lr', 'n_layers', 'hid_dim', 'epochs', 'clip', 'clip_node', 'trim_rule', 'ns', 'debug', 'device', 
                'sampling_rate', 'optimizer', 'att_mode', 'att_submode', 'att_layers', 'att_hid_dim', 'att_lr', 'att_batch_size',
                'att_epochs', 'sha_lr', 'sha_epochs', 'sha_ratio']
    else:
        keys = ['mode', 'seed', 'performance_metric', 'dataset', 'batch_size', 'n_neighbor', 'model_type', 
                'lr', 'n_layers', 'hid_dim', 'epochs', 'debug', 'device', 'optimizer', 'att_mode', 'att_submode', 
                'att_layers', 'att_hid_dim', 'att_lr', 'att_batch_size', 'att_epochs', 'sha_lr', 'sha_epochs', 'sha_ratio']
        
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

    if args.att_mode == 'blackbox':
        att_key = ['att_mode', 'att_submode', 'att_layers', 'att_hid_dim', 'att_lr', 'att_batch_size', 
                'att_epochs', 'sha_lr', 'sha_epochs', 'sha_ratio']
    else:
        att_key = ['att_mode', 'att_submode', 'att_layers', 'att_hid_dim', 'att_lr', 'att_batch_size',
                    'att_epochs', 'sha_ratio']

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
    
    att_str = ''
    for key in att_key:
        att_str += f"{key}_{getattr(args, key)}_"


    name = {
        'data': data_str[:-1],
        'model': model_str[:-1],
        'dp': dp_str[:-1],
        'att': att_str[:-1],
        'general': data_str + model_str + dp_str + att_str + date_str
    }

    return name