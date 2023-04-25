import torch
import os
import numpy as np
from copy import deepcopy
from multiprocessing import Pool

def random_trimming(roots, k, current_node, appear_dict=None, model=None, graph=None):
    r = deepcopy(roots)
    if current_node in r: r.remove(current_node)
    r = np.array(r)
    list_of_root = np.random.choice(a=r, size=k, replace=False).astype(int)
    return list_of_root.tolist()

def adhoc_trimming_rank(roots, k, current_node, appear_dict=None, model=None, graph=None):
    r = deepcopy(roots)
    if current_node in r: r.remove(current_node)
    rank_at_r = []
    for n in r: rank_at_r.append(np.sum(np.array(appear_dict.node_dict[current_node].root_dict[n]['rank']) + 1))
    root_rank = sorted(list(zip(r, rank_at_r)), key=lambda x: x[1])
    list_of_root = [x[0] for x in root_rank]
    return list_of_root[:k]

def impact_aware_trimming(roots, k, current_node, appear_dict=None, model=None, graph=None):
    model.eval()
    r = deepcopy(roots)
    appear_dict_ = deepcopy(appear_dict)
    if current_node in r: r.remove(current_node)
    args = [(r_, current_node, appear_dict, appear_dict_, model, graph) for r_ in r]
    with torch.no_grad():
        print(f'Working with {os.cpu_count()} threads, for {len(r)} tasks')
        pool = Pool(processes=os.cpu_count())
        results = pool.map(get_val, args)
        pool.close()
        smape = [res for res in results]
    root_rank = sorted(smape, key=lambda x: x[1])
    list_of_root = [x[0] for x in root_rank]
    return list_of_root[:k]

def get_val(args):
    root, curr_node, appear_dict, appear_dict_, model, graph = args
    print(f'working with root {root}')
    appear_dict_.trim_node_from_root(node_id=curr_node, root=root)
    blocks = appear_dict.build_blocks(root=root, graph=graph)
    blocks_ = appear_dict_.build_blocks(root=root, graph=graph)
    inputs = blocks[0].srcdata['feat']
    inputs_ = blocks_[0].srcdata['feat']
    predictions = model(blocks, inputs)
    predictions_ = model(blocks_, inputs_)
    val = torch.norm(predictions - predictions_, p=1) / (torch.norm(predictions, p=1) + torch.norm(predictions_, p=1))
    return (root, val.item())