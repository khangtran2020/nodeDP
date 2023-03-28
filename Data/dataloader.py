import dgl
import torch
import logging
import numpy as np
from torch import device
from typing import Tuple, Dict, List, Optional, Union, Set
from dgl.dataloading.base import NID, EID
from dgl.dataloading import transforms, DataLoader


class ComputeSubgraphSampler(dgl.dataloading.BlockSampler):
    def __init__(self, num_neighbors, device='cpu'):
        super().__init__(len(num_neighbors))
        self.num_layers = len(num_neighbors)
        self.fanouts = num_neighbors
        self.device = device

    def sample_frontier(self, block_id, g, seed_nodes):
        fanout = self.fanouts[block_id]
        if fanout is None:
            frontier = dgl.in_subgraph(g, seed_nodes)
        else:
            frontier = dgl.sampling.sample_neighbors(g, seed_nodes, fanout)
        return frontier

    def sample_blocks(self, g, seed_nodes, exclude_eids=None):
        output_nodes = seed_nodes
        blocks = []
        for fanout in reversed(self.fanouts):
            frontier = g.sample_neighbors(
                seed_nodes, fanout, output_device=self.device)
            eid = frontier.edata[EID]
            block = transforms.to_block(frontier, seed_nodes, include_dst_in_src=False)
            block.edata[EID] = eid
            seed_nodes = block.srcdata[NID]
            blocks.insert(0, block)

        return blocks

    def sample(self, g, seed_nodes, exclude_eids=None):
        sub_graph = {}
        for node in seed_nodes:
            blocks = self.sample_blocks(g, seed_nodes=[node.item()])
            sub_graph[node.item()] = blocks
        return sub_graph


class NodeDataLoader(object):
    logger = logging.getLogger('graph-dl')

    def __init__(self, g: dgl.DGLGraph, batch_size: int, shuffle: bool, num_workers: int,
                 num_nodes: Union[int, List], cache_result: bool = False, drop_last: bool = True, mode: str = 'train',
                 device = 'cpu'):

        self.num_nodes = num_nodes
        self.g = g
        self.batch_size = batch_size
        self.n_batch = int(len(g.nodes().tolist()) / self.batch_size) if drop_last else int(
            (len(g.nodes().tolist()) + self.batch_size) / self.batch_size)
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.cache_result = cache_result
        self.cache = None
        self.mode = mode
        self.device = device
        self.drop_last = drop_last

    def __iter__(self):
        yield from self.iter_sage()

    def iter_sage(self):
        if self.cache_result and self.cache and len(self.cache) == len(self):
            self.logger.info('DL loaded from cache')
            for e in self.cache:
                yield e
        else:
            from torch.utils.data import DataLoader
            g = self.g
            seeds = torch.from_numpy(self.sample_seeds())
            sampler = ComputeSubgraphSampler(num_neighbors=self.num_nodes, device=self.device)
            bz = self.batch_size
            dl = DataLoader(seeds, batch_size=bz, shuffle=self.shuffle, drop_last=self.drop_last)

            if self.cache_result:
                self.cache = []
            for seed in dl:
                sub_graph = sampler.sample(g=g, seed_nodes=seed)
                encoded_seeds = seed.numpy()
                yield encoded_seeds, sub_graph

    def sample_seeds(self) -> List:
        g = self.g
        list_of_nodes = torch.index_select(g.nodes(), 0, g.ndata['train_mask'].nonzero().squeeze()).numpy()
        seeds = np.random.choice(list_of_nodes, self.batch_size, replace=len(list_of_nodes) < self.batch_size)
        return seeds.astype(int)

    def __len__(self):
        return self.n_batch
