import dgl
import sys
import torch
import numpy as np
from rich import print as rprint
from Utils.utils import get_index_bynot_value, get_index_by_list, get_index_by_value, timeit
from dgl.dataloading import to_block
from loguru import logger
from copy import deepcopy

# logger.remove()
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")

def remove_substring_from_string(a, b):
    ls = a.split('|')
    ls.remove(b)
    return '|'.join(ls)


class AppearDict(object):

    def __init__(self, roots, subgraph, graph, clip_node, rule, num_layer, debug, step, device, model):
        self.debug = debug
        self.num_layer = num_layer
        self.roots = roots
        self.batch_size = len(roots)
        self.subgraph = subgraph
        self.graph = graph
        self.clip_node = clip_node
        self.step = step
        self.device = device
        self.rule = rule
        self.model = model
        self.num_node = self.graph.nodes().size(dim=0)
        self.node_appear = np.zeros(self.num_node).astype(int)
        self.node_roots = np.array(["" for i in range(self.num_node)], dtype='object')
        self.root_dict = {}
        self.build_dict()
        self.rm_substr = np.vectorize(remove_substring_from_string)
        self.trim_info = {
            'num_subgraphs': len(roots),
            'num_subgraphs_trimmed': 0,
            'trimmed_subgraphs': []
        }
        # rprint(f"Orginal roots: {self.roots}")
        if len(self.subgraph.keys()) > self.batch_size:
            logger.error("ROOT as string problem")
            sys.exit()
        logger.info(f"Batch size {self.batch_size}")


    def build_dict(self):

        for i, root in enumerate(self.roots):

            nodes = torch.Tensor([]).to(self.device)
            key = root.item()
            blocks = self.subgraph[key]
            for block in blocks:
                src_node = block.srcdata[dgl.NID]
                dst_node = block.dstdata[dgl.NID]
                nodes = torch.cat((nodes, src_node,dst_node), dim=0).unique().int()
            index = nodes.tolist()
            self.node_appear[index] += 1
            self.node_roots[index] += f'{key}|'
            self.root_dict[key] = index
        self.node_roots = np.char.strip(self.node_roots.astype(str), '|')
        self.num_node_batch = len(np.where(self.node_appear > 0)[0].tolist())
        self.node_to_trim = np.where(self.node_appear > self.clip_node)[0].tolist()
        rprint(f"% of node to trim: {len(self.node_to_trim)/self.num_node_batch*100:.2f}")

    def trim(self):
        if self.debug:
            root_node = deepcopy(self.node_roots)
        logger.info(f"BEGIN TRIMMING FOR STEP {self.step}\n\n")
        idx = np.argmax(self.node_appear)
        # rprint("Type of idx:", type(idx))
        val = self.node_appear[idx]
        while (val > self.clip_node):
            # rprint(f'\n\nTrimming node {idx}, appeard {self.node_appear[idx]} times with roots {self.node_roots[idx]}')
            # get list root
            roots_str = np.array(self.node_roots[idx].split('|'))
            # rprint(f"# root need to be trimmed of node {idx} org: {len(roots_str) - self.clip_node}")
            if f'{idx}' in roots_str:
                roots_str = np.delete(roots_str, axis=0, obj=np.where(roots_str == f'{idx}')[0].tolist())
            if self.rule == 'random':
                root_to_trim = np.random.choice(a=roots_str, size=self.node_appear[idx] - self.clip_node, replace=False)
            elif self.rule == 'adhoc':
                # rprint(f"Root of node {idx} at selecting: {roots_str}")
                ranks = [(root, self.get_rank_of_node_in_root(node_id=idx, root=int(root))) for root in roots_str]
                ranks = sorted(ranks, key=lambda x: x[1])
                root_to_trim = [x[0] for x in ranks[:int(self.node_appear[idx] - self.clip_node)]]
            else:
                with timeit(logger=logger, task=f'getting-impact-rank-node-{idx}'):
                    ranks = [(root, self.get_grad_in_root(root=int(root), node=idx)) for root in roots_str]
                    ranks = sorted(ranks, key=lambda x: x[1])
                    root_to_trim = [x[0] for x in ranks[:int(self.node_appear[idx] - self.clip_node)]]

            for root in root_to_trim:
                if int(root) not in self.trim_info['trimmed_subgraphs']:
                    self.trim_info['trimmed_subgraphs'].append(int(root))
                    self.trim_info['num_subgraphs_trimmed'] += 1
                    self.trim_info[int(root)] = {
                        'num_node_org': len(self.root_dict[int(root)]),
                        'num_node_trimmed': 0
                    }
                blocks = deepcopy(self.subgraph[int(root)])
                dst_n = torch.Tensor([int(root)]).int()
                # rprint(f"Current dst_n: {dst_n}")
                trimmed = False
                new_blocks = []
                for i in reversed(range(self.num_layer)):
                    block = blocks[i]
                    if trimmed:
                        src_node = block.srcdata[dgl.NID]
                        dst_node = block.dstdata[dgl.NID]
                        src_edge, dst_edge = block.edges()
                        src_edge = torch.index_select(input=src_node, dim=0, index=src_edge)
                        dst_edge = torch.index_select(input=dst_node, dim=0, index=dst_edge)
                        indices = get_index_by_list(dst_edge, test_arr=dst_n)
                        dst_edge = torch.index_select(input=dst_edge, dim=0, index=indices).int()
                        src_edge = torch.index_select(input=src_edge, dim=0, index=indices).int()
                        g = dgl.graph((src_edge, dst_edge), num_nodes=self.num_node)
                        block = to_block(g, dst_nodes=dst_n, include_dst_in_src=False)

                    if idx in block.srcdata[dgl.NID]:
                        src_node = block.srcdata[dgl.NID]
                        dst_node = block.dstdata[dgl.NID]
                        src_edge, dst_edge = block.edges()
                        src_edge = torch.index_select(input=src_node, dim=0, index=src_edge)
                        dst_edge = torch.index_select(input=dst_node, dim=0, index=dst_edge)
                        indices = get_index_bynot_value(a=src_edge, val=idx)
                        dst_edge = torch.index_select(input=dst_edge, dim=0, index=indices).int()
                        src_edge = torch.index_select(input=src_edge, dim=0, index=indices).int()
                        g = dgl.graph((src_edge, dst_edge), num_nodes=self.num_node)
                        block = to_block(g, dst_nodes=dst_n, include_dst_in_src=False)
                        trimmed = True


                    dst_n = block.srcdata[dgl.NID].int()
                    new_blocks.insert(0, block)

                nodes_new = torch.Tensor([]).to(self.device)
                for block in new_blocks:
                    src_node = block.srcdata[dgl.NID]
                    dst_node = block.dstdata[dgl.NID]
                    nodes_new = torch.cat((nodes_new, src_node, dst_node), dim=0).unique().int()

                if idx in nodes_new:
                    logger.error(f"Deleted fails since the targeted still in subgraph {root}")
                    sys.exit()
                nodes_new = set(nodes_new.tolist())
                nodes_old = set(self.root_dict[int(root)])
                node_removed = list(nodes_old.symmetric_difference(nodes_new))
                self.trim_info[int(root)]['num_node_trimmed'] += len(node_removed)
                # rprint(f"At root {root}\nNode removed: {node_removed}")
                if len(node_removed) > 0:
                    self.root_dict[int(root)] = list(nodes_new)
                    self.node_appear[node_removed] -= 1
                    self.node_roots[node_removed] = self.rm_substr(self.node_roots[node_removed], root)
                    self.subgraph[int(root)] = new_blocks

            # update root for node idx
            # rprint("Len after updated:", len(self.node_roots[idx].split('|')), 'in roots:', self.node_roots[idx].split('|'))
            idx = np.argmax(self.node_appear)
            val = self.node_appear[idx]
        if self.debug:
            if len(self.subgraph.keys()) > self.batch_size:
                logger.error("ROOT as string problem")
                # rprint(f"New roots: {list(self.subgraph.keys())}")
                sys.exit()
            self.check_node()
        logger.info(f"\n\nDONE TRIMMING FOR STEP {self.step}")
        return self.trim_info

    def check_node(self):

        nodes_appear = np.zeros(self.num_node).astype(int)
        nodes_root = np.array(["" for i in range(self.num_node)], dtype='object')
        for i, root in enumerate(self.roots):
            key = root.item()
            nodes = torch.Tensor([]).to(self.device)
            blocks = self.subgraph[key]
            for block in blocks:
                src_node = block.srcdata[dgl.NID]
                dst_node = block.dstdata[dgl.NID]
                nodes = torch.cat((nodes, src_node,dst_node), dim=0).unique().int()
            index = nodes.tolist()
            nodes_appear[index] += 1
            nodes_root[index] += f'{key}|'

        node_to_trim = np.where(nodes_appear > self.clip_node)[0].tolist()
        if np.max(nodes_appear) > self.clip_node:
            logger.error(f"DID NOT PASS CHECK AT STEP {self.step}")
            rprint("========== DETAILS =========\n")
            for node in node_to_trim:
                rprint(f"Node {node} still appears {nodes_appear[node]} at root {nodes_root[node]}")
            rprint("\n========== DETAILS =========")
            sys.exit()
        else:
            logger.info(f"PASSED CHECK FOR STEP {self.step}")

    def joint_blocks(self):
        new_blocks = []
        dst_n = torch.Tensor(self.roots).int()
        for i in reversed(range(self.num_layer)):
            src_edge = torch.Tensor([]).to(self.device).int()
            dst_edge = torch.Tensor([]).to(self.device).int()
            for root in self.roots:
                key = root.item()
                block = self.subgraph[key][i]
                src_node = block.srcdata[dgl.NID]
                dst_node = block.dstdata[dgl.NID]
                src_ed, dst_ed = block.edges()
                src_edge = torch.cat((src_edge, torch.index_select(src_node, 0, src_ed).int()), dim=0)
                dst_edge = torch.cat((dst_edge, torch.index_select(dst_node, 0, dst_ed).int()), dim=0)
            src_edge = src_edge.int()
            dst_edge = dst_edge.int()
            g = dgl.graph((src_edge, dst_edge), num_nodes=self.num_node)
            g.ndata['feat'] = self.graph.ndata['feat'].clone()
            g.ndata['label'] = self.graph.ndata['label'].clone()
            blk = to_block(g=g, dst_nodes=dst_n, include_dst_in_src=True)
            dst_n = blk.srcdata[dgl.NID]
            new_blocks.insert(0, blk)
        return new_blocks

    def build_block(self, root):
        new_blocks = []
        dst_n = torch.Tensor([root]).int()
        for i in reversed(range(self.num_layer)):
            block = self.subgraph[root][i]
            src_node = block.srcdata[dgl.NID]
            dst_node = block.dstdata[dgl.NID]
            src_ed, dst_ed = block.edges()
            src_edge = torch.index_select(src_node, 0, src_ed).int()
            dst_edge = torch.index_select(dst_node, 0, dst_ed).int()
            g = dgl.graph((src_edge, dst_edge), num_nodes=self.num_node)
            g.ndata['feat'] = self.graph.ndata['feat'].clone()
            g.ndata['label'] = self.graph.ndata['label'].clone()
            blk = to_block(g=g, dst_nodes=dst_n, include_dst_in_src=True)
            dst_n = blk.srcdata[dgl.NID]
            new_blocks.insert(0, blk)
        return new_blocks

    def get_grad_in_root(self, root, node):
        blocks = self.build_block(root=root)
        nodes = blocks[0].srcdata[dgl.NID]
        index = get_index_by_value(a=nodes, val=node)
        inputs = torch.autograd.Variable(blocks[0].srcdata["feat"], requires_grad=True)
        predictions = self.model(blocks, inputs)
        predictions.sum().backward()
        return inputs.grad[index].norm(p=2)


    def get_rank_of_node_in_root(self, node_id, root):
        blocks = self.subgraph[root]
        rank = 0
        for i in reversed(range(self.num_layer)):
            block = blocks[i]
            if node_id in block.srcdata[dgl.NID]:
                rank += i+1
        return rank