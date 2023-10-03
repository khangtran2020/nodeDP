import dgl
import logging
from torch.utils.data import DataLoader as DL
from typing import List, Union
from dgl.dataloading.base import NID, EID
from dgl.dataloading import transforms, DataLoader


class ComputeSubgraphSampler(dgl.dataloading.BlockSampler):
    def __init__(self, num_neighbors, device):
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
                 device='cpu', sampling_rate=0.2):

        self.num_nodes = num_nodes
        self.g = g
        self.batch_size = batch_size
        if len(g.nodes().tolist()) % self.batch_size == 0:
            self.n_batch = int(len(g.nodes().tolist()) / self.batch_size)
        else:
            if drop_last:
                self.n_batch = int(len(g.nodes().tolist()) / self.batch_size)
            else:
                self.n_batch = int((len(g.nodes().tolist()) + self.batch_size) / self.batch_size)
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.cache_result = cache_result
        self.cache = None
        self.mode = mode
        self.device = device
        self.drop_last = drop_last
        self.sampling_rate = sampling_rate
        self.seeds = g.nodes()

    def __iter__(self):
        g = self.g
        sampler = ComputeSubgraphSampler(num_neighbors=self.num_nodes, device=self.device)
        bz = self.batch_size
        dl = DL(self.seeds, batch_size=bz, shuffle=self.shuffle, drop_last=self.drop_last)
        for seed in dl:
            sub_graph = sampler.sample(g=g, seed_nodes=seed)
            encoded_seeds = seed
            yield encoded_seeds, sub_graph

    def __len__(self):
        return self.n_batch
