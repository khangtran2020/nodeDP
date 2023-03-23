import dgl
import torch
from dgl.dataloading import transforms

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
        graph_device = g.device
        subgraph = {}
        for node in seed_nodes.item():
            blocks = []
            for block_id in reversed(range(self.num_layers)):
                seed_nodes_in = torch.LongTensor([node])
                if isinstance(seed_nodes_in, dict):
                    seed_nodes_in = {ntype: nodes.to(graph_device) \
                                     for ntype, nodes in seed_nodes_in.items()}
                else:
                    seed_nodes_in = seed_nodes_in.to(graph_device)
                frontier = self.sample_frontier(block_id, g, seed_nodes_in)

                if self.output_device is not None:
                    frontier = frontier.to(self.output_device)
                    if isinstance(seed_nodes, dict):
                        seed_nodes_out = {ntype: nodes.to(self.output_device) \
                                          for ntype, nodes in seed_nodes.items()}
                    else:
                        seed_nodes_out = seed_nodes.to(self.output_device)
                else:
                    seed_nodes_out = seed_nodes

                block = transforms.to_block(frontier, seed_nodes_out)

                seed_nodes = {ntype: block.srcnodes[ntype].data[dgl.NID] for ntype in block.srctypes}
                blocks.insert(0, block)
            subgraph[node] = blocks
        return subgraph