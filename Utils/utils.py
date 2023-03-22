import random
import os
import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def get_name(args, current_date):
    dataset_str = f'{args.dataset}_{args.ratio}_'
    date_str = f'{current_date.day}-{current_date.month}-{current_date.year}_{current_date.hour}-{current_date.minute}-{current_date.second}'
    model_str = f'{args.mode}_{args.epochs}_{args.performance_metric}_{args.optimizer}_'
    dp_str = f'{args.clip}_{args.ns}_'
    if args.mode == 'clean':
        res_str = dataset_str + model_str + date_str
    elif args.mode == 'dp':
        res_str = dataset_str + model_str + dp_str + date_str
    return res_str
