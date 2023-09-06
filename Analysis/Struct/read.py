import os
import sys
import dgl
import pandas as pd
import torch
import numpy as np
import networkx as nx
from functools import partial
from Data.facebook import Facebook
from Data.amazon import Amazon
from copy import deepcopy
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from Data.dataloader import NodeDataLoader
from ogb.nodeproppred import DglNodePropPredDataset
from Utils.utils import *
from dgl.data import CoraGraphDataset, CiteseerGraphDataset, PubmedGraphDataset
from torch_geometric.transforms import Compose, RandomNodeSplit
from joblib import Parallel, delayed
from loguru import logger
from Utils.utils import timeit
from Data.read import node_split, filter_class_by_count, reduce_desity, graph_split

logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")

def read_data(args, history):
    if args.seed_info == '':
        graph, list_of_label = get_dataset(data_name=args.dataset, file=None)
        history['tr_id'] = get_index_bynot_value(a=graph.ndata['train_mask'], val=0)
        history['va_id'] = get_index_bynot_value(a=graph.ndata['val_mask'], val=0)
        history['te_id'] = get_index_bynot_value(a=graph.ndata['test_mask'], val=0)
    else:
        graph, list_of_label = get_dataset(data_name=args.dataset, file=args.res_path+args.seed_info)
    args.num_class = len(list_of_label)
    args.num_feat = graph.ndata['feat'].shape[1]
    graph = dgl.remove_self_loop(graph)
    graph_ = reduce_desity(g=graph, dens_reduction=args.density)

    g_train, g_val, g_test = graph_split(graph=graph, drop=True)

    if (args.submode == 'density') and (args.density == 1.0):
        g_train_, g_val_, g_test_ = graph_split(graph=graph_, drop=False)
    else:
        g_train_, g_val_, g_test_ = graph_split(graph=graph_, drop=True)
    
    org_graph_info = (g_train, g_val, g_test)
    drop_graph_info = (g_train_, g_val_, g_test_)

    return org_graph_info, drop_graph_info
    


def get_dataset(data_name, file=None):
    if data_name == 'reddit':
        data = dgl.data.RedditDataset()
        graph = data[0]
        node_split_seeded(graph=graph, val_size=0.1, test_size=0.15, file=file)
        list_of_label = filter_class_by_count(graph=graph, min_count=10000)
    elif data_name == 'cora':
        data = CoraGraphDataset()
        graph = data[0]
        del(graph.ndata['train_mask'])
        del(graph.ndata['val_mask'])
        del(graph.ndata['test_mask'])
        node_split_seeded(graph=graph, val_size=0.1, test_size=0.15, file=file)
        list_of_label = filter_class_by_count(graph=graph, min_count=0)
    elif data_name == 'citeseer':
        data = CiteseerGraphDataset()
        graph = data[0]
        del(graph.ndata['train_mask'])
        del(graph.ndata['val_mask'])
        del(graph.ndata['test_mask'])
        node_split_seeded(graph=graph, val_size=0.1, test_size=0.15, file=file)
        list_of_label = filter_class_by_count(graph=graph, min_count=0)
    elif data_name == 'pubmed':
        data = PubmedGraphDataset()
        graph = data[0]
        del(graph.ndata['train_mask'])
        del(graph.ndata['val_mask'])
        del(graph.ndata['test_mask'])
        node_split_seeded(graph=graph, val_size=0.1, test_size=0.15, file=file)
        list_of_label = filter_class_by_count(graph=graph, min_count=0)
    elif data_name == 'facebook':
        load_data = partial(Facebook, name='UIllinois20', target='year',
                            transform=Compose([
                                RandomNodeSplit(num_val=0.1, num_test=0.15)
                                # FilterClassByCount(min_count=1000, remove_unlabeled=True)
                            ])
                            )
        data = load_data(root='dataset/')[0]
        src_edge = data.edge_index[0]
        dst_edge = data.edge_index[1]
        graph = dgl.graph((src_edge, dst_edge), num_nodes=data.x.size(dim=0))
        graph.ndata['feat'] = data.x
        graph.ndata['label'] = data.y
        graph.ndata['train_mask'] = data.train_mask
        graph.ndata['val_mask'] = data.val_mask
        graph.ndata['test_mask'] = data.test_mask
        node_split_seeded(graph=graph, val_size=0.1, test_size=0.15, file=file)
        list_of_label = filter_class_by_count(graph=graph, min_count=1000)
        # sys.exit()
    elif data_name == 'amazon':
        load_data = partial(Amazon,
                            transform=Compose([
                                RandomNodeSplit(num_val=0.1, num_test=0.15)
                            ])
                            )
        data = load_data(root='dataset/')[0]
        src_edge = data.edge_index[0]
        dst_edge = data.edge_index[1]
        graph = dgl.graph((src_edge, dst_edge), num_nodes=data.x.size(dim=0))
        graph.ndata['feat'] = data.x
        graph.ndata['label'] = data.y
        graph.ndata['train_mask'] = data.train_mask
        graph.ndata['val_mask'] = data.val_mask
        graph.ndata['test_mask'] = data.test_mask
        node_split_seeded(graph=graph, val_size=0.1, test_size=0.15, file=file)
        list_of_label = filter_class_by_count(graph=graph, min_count=6000)
    return graph, list_of_label

def node_split_seeded(graph, val_size=0.1, test_size=0.15, file=None):
    if file == None:
        node_split(graph=graph, val_size=val_size, test_size=test_size)
    else:
        with open(file=file, mode='rb') as f:
            history = pickle.load(f)
        tr_id = history['tr_id']
        va_id = history['va_id']
        te_id = history['te_id']
        
        tr_mask = torch.zeros(graph.nodes().size(dim=0)).int()
        tr_mask[tr_id.int()] = 1

        va_mask = torch.zeros(graph.nodes().size(dim=0)).int()
        va_mask[va_id.int()] = 1

        te_mask = torch.zeros(graph.nodes().size(dim=0)).int()
        te_mask[te_id.int()] = 1

        graph.ndata['train_mask'] = tr_mask
        graph.ndata['val_mask'] = va_mask
        graph.ndata['test_mask'] = te_mask

