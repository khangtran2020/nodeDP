import dgl
import sys
import torch
import numpy as np
from rich import print as rprint
from rich.pretty import pretty_repr
from Utils.utils import get_index_by_value, get_index_bynot_value
from functools import partial
from dgl.dataloading import to_block
from loguru import logger
from copy import deepcopy
from dgl import NID

# logger.remove()
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")


class Node(object):

    def __init__(self, node_id):
        self.id = node_id
        self.num_tree = 0
        self.roots = []
        self.root_dict = {}

    def add_sub_graph(self, root, rank):
        if root in self.roots:
            if rank not in self.root_dict[root]['rank']:
                self.root_dict[root]['num_appear'] += 1
                self.root_dict[root]['rank'].append(rank)
        else:
            self.num_tree += 1
            self.roots.append(root)
            self.root_dict[root] = {
                'num_appear': 1,
                'rank': [rank]
            }


class AppearDict(object):

    def __init__(self, roots, subgraphs, id=0, model=None, graph=None, k=2, num_layer=2,
                 debug=False):
        self.roots = roots
        self.roots_index = dict(zip(roots, range(len(roots))))
        self.id = id
        self.subgraphs = subgraphs
        self.graph = graph
        self.node_dict, self.root_dict = self.build_node_dict(roots, subgraphs)
        self.k = k
        self.num_layer = num_layer
        self.model = model
        self.node_to_trim = self.find_node_to_trim()
        if len(self.node_to_trim) > 0:
            self.need_to_trim = True
            original_block = self.joint_blocks()
            orginal_pred = self.get_prediction(blocks=original_block)
            self.trim_info = {
                '# subgraphs': len(roots),
                '# subgraphs trimmed': 0,
                'trimmed subgraphs': []
            }
            self.trim_node = partial(trim_node, org_pred=orginal_pred, k=self.k)
            self.debug = debug
            if self.debug:
                self.copy = AppearDict(roots=roots, subgraphs=subgraphs, id=1, model=model, graph=graph, k=k,
                                       num_layer=num_layer, debug=False)
        else:
            self.need_to_trim = False

    def find_node_to_trim(self):
        node_appear_dict = {}
        for node in self.node_dict.keys():
            if self.node_dict[node].num_tree > self.k:
                node_appear_dict[node] = self.node_dict[node].num_tree
        return node_appear_dict

    def build_node_dict(self, roots, subgraphs):
        results = {}
        root_dict_ = {}
        for root in roots:
            root_dict = {
                'nodes': [],
                '# nodes org': 0,
                '# edges org': 0
            }
            blocks = subgraphs[root]
            for i, block in enumerate(blocks):
                ans_rank = i + 1
                child_rank = i
                src_node = block.srcdata[dgl.NID]
                dst_node = block.dstdata[dgl.NID]
                src_edge, dst_edge = block.edges()
                for j, n in enumerate(dst_node.tolist()):
                    if n not in results.keys(): results[n] = Node(node_id=n)
                    if n not in root_dict['nodes']: root_dict['nodes'].append(n)
                    indices = get_index_by_value(a=dst_edge, val=j)
                    child_idx = torch.index_select(src_edge, 0, indices).unique()
                    child = torch.index_select(src_node, 0, child_idx).tolist()
                    results[n].add_sub_graph(root=root, rank=ans_rank)
                    root_dict['# edges org'] += len(child)
                for j, n in enumerate(src_node.tolist()):
                    if n not in results.keys(): results[n] = Node(node_id=n)
                    if n not in root_dict['nodes']: root_dict['nodes'].append(n)
                    indices = get_index_by_value(a=src_edge, val=j)
                    ans_idx = torch.index_select(dst_edge, 0, indices).unique()
                    ancestor = torch.index_select(dst_node, 0, ans_idx).tolist()
                    results[n].add_sub_graph(root=root, rank=child_rank)
                    root_dict['# edges org'] += len(ancestor)
                root_dict['# nodes org'] = len(root_dict['nodes'])
                root_dict['graph'] = self.graph.subgraph(torch.LongTensor(root_dict['nodes']))
                root_dict_[root] = root_dict
        return results, root_dict_

    def build_root_dict(self, root):
        blocks = self.subgraphs[root]
        root_dict = {
            'root': root
        }
        for i, block in enumerate(blocks):
            rank_child = i
            rank_ans = i + 1
            if f'rank_{rank_child}' not in root_dict.keys(): root_dict[f'rank_{rank_child}'] = {}
            if f'rank_{rank_ans}' not in root_dict.keys(): root_dict[f'rank_{rank_ans}'] = {}
            src_node = block.srcdata[dgl.NID]
            dst_node = block.dstdata[dgl.NID]
            src_edge, dst_edge = block.edges()
            for j, n in enumerate(dst_node.tolist()):
                if n not in root_dict[f'rank_{rank_ans}'].keys(): root_dict[f'rank_{rank_ans}'][n] = {'child': [],
                                                                                                      'ans': []}
                indices = get_index_by_value(a=dst_edge, val=j)
                child_idx = torch.index_select(src_edge, 0, indices).unique()
                child = torch.index_select(src_node, 0, child_idx).tolist()
                root_dict[f'rank_{rank_ans}'][n]['child'] += child
            for j, n in enumerate(src_node.tolist()):
                if n not in root_dict[f'rank_{rank_child}'].keys(): root_dict[f'rank_{rank_child}'][n] = {'child': [],
                                                                                                          'ans': []}
                indices = get_index_by_value(a=src_edge, val=j)
                ans_idx = torch.index_select(dst_edge, 0, indices).unique()
                ancestor = torch.index_select(dst_node, 0, ans_idx).tolist()
                root_dict[f'rank_{rank_child}'][n]['ans'] += ancestor
        return root_dict

    def getkeys(self):
        return self.node_dict.keys()

    def trim(self):
        for node, val in self.node_to_trim.items():
            self.trim_node(node, appear_dict=self)

        self.trim_info['% subgraph'] = self.trim_info['# subgraphs trimmed'] / self.trim_info['# subgraphs']
        self.trim_info['% node avg'] = []
        self.trim_info['% edge avg'] = []
        self.trim_info['avg rank'] = []
        for root in self.trim_info['trimmed subgraphs']:
            self.trim_info[root]['% node'] = self.trim_info[root]['# nodes trimmed'] / \
                                             self.root_dict[root]['# nodes org']
            self.trim_info['% node avg'].append(self.trim_info[root]['% node'])
            self.trim_info[root]['% edge'] = self.trim_info[root]['# edges trimmed'] / \
                                             self.root_dict[root]['# edges org']
            self.trim_info['% edge avg'].append(self.trim_info[root]['% edge'])
            self.trim_info[root]['average rank trimmed'] = sum(self.trim_info[root]['rank trimmed']) / \
                                                           len(self.trim_info[root]['rank trimmed'])
            self.trim_info['avg rank'].append(self.trim_info[root]['average rank trimmed'])
        self.trim_info['% node avg'] = sum(self.trim_info['% node avg']) / len(self.trim_info['% node avg'])
        self.trim_info['% edge avg'] = sum(self.trim_info['% edge avg']) / len(self.trim_info['% edge avg'])
        self.trim_info['avg rank'] = sum(self.trim_info['avg rank']) / len(self.trim_info['avg rank'])

        if self.debug:
            node_dict, root_dict = self.build_node_dict(self.roots, self.subgraphs)
            self.check_nodes(node_dict)
        return self.trim_info

    def build_blocks(self, graph):
        new_blocks = []
        dst_n = torch.Tensor(self.roots).int()
        for i in reversed(range(self.num_layer)):
            src_edge = np.array([])
            dst_edge = np.array([])
            for root in self.roots:
                # root_dict = self.root_dict[root]
                for item in self.root_dict[root][f'rank_{i}'].items():
                    val = np.array(item[1]['ans']).astype(int)
                    temp = val - item[0]
                    src_edge = np.concatenate((src_edge, val - temp), axis=0)
                    dst_edge = np.concatenate((dst_edge, temp + item[0]), axis=0)

            src_edge = torch.Tensor(src_edge).int()
            dst_edge = torch.Tensor(dst_edge).int()
            g = dgl.graph((src_edge, dst_edge), num_nodes=len(graph.nodes()))
            g.ndata['feat'] = graph.ndata['feat'].clone()
            g.ndata['label'] = graph.ndata['label'].clone()
            blk = to_block(g=g, dst_nodes=dst_n, include_dst_in_src=True)
            dst_n = blk.srcdata[dgl.NID]
            new_blocks.insert(0, blk)
        return new_blocks

    def joint_blocks(self):
        new_blocks = []
        dst_n = torch.Tensor(self.roots).int()
        for i in reversed(range(self.num_layer)):
            src_edge = torch.Tensor([])
            dst_edge = torch.Tensor([])
            for root in self.roots:
                block = self.subgraphs[root][i]
                src_node = block.srcdata[NID]
                dst_node = block.dstdata[NID]
                src_ed, dst_ed = block.edges()
                src_edge = torch.cat((src_edge, torch.index_select(src_node, 0, src_ed)), dim=0)
                dst_edge = torch.cat((dst_edge, torch.index_select(dst_node, 0, dst_ed)), dim=0)
            src_edge = src_edge.int()
            dst_edge = dst_edge.int()
            g = dgl.graph((src_edge, dst_edge), num_nodes=len(self.graph.nodes()))
            g.ndata['feat'] = self.graph.ndata['feat'].clone()
            g.ndata['label'] = self.graph.ndata['label'].clone()
            blk = to_block(g=g, dst_nodes=dst_n, include_dst_in_src=True)
            dst_n = blk.srcdata[dgl.NID]
            new_blocks.insert(0, blk)
        return new_blocks

    def get_prediction(self, blocks):
        inputs = blocks[0].srcdata['feat']
        predictions = self.model(blocks, inputs)
        return predictions

    def print_nodes(self):
        node_dict = {}
        for node in self.node_dict.keys():
            node_dict[node] = self.node_dict[node].root_dict
        rprint('Node dict: \n', pretty_repr(node_dict))

    def check_nodes(self, node_dict):
        ok = True
        for node in node_dict.keys():
            if node_dict[node].num_tree > self.k:
                logger.error(f'Node {node} still appears {node_dict[node].num_tree} times:')
                for root in node_dict[node].roots:
                    rprint(f'Root {root} at rank: {node_dict[node].root_dict[root]["rank"]}')

                # sys.exit('ERROR: You stupid ass bitch')
                ok = False
        if ok:
            logger.info(f'Everything is great in this step')
        else:
            sys.exit('ERROR: You stupid ass bitch')

    def handle_error(self, root, node, rank):
        logger.info(f'=============== ERROR node {node} at rank {rank} in root {root} ===============')
        if self.debug:
            org_root_dict = self.copy.build_root_dict(root=root)
            logger.info(f'Original dict of root {root}: \n {pretty_repr(org_root_dict)}')

            for i, item in enumerate(self.deleted_node[root]):
                _ = self.copy.remove_node_from_root_at_rank(node_id=deepcopy(item[0]), root=deepcopy(root),
                                                            rank=deepcopy(item[1]), queue=None)
                org_root_dict = self.copy.build_root_dict(root=root)
                logger.info(f'Dict of root {root} at step {i} with item {item}: \n {pretty_repr(org_root_dict)}')
        else:
            logger.info(f'Turn on debug you motherfucker lazy bitch')
        logger.info(f'==============================  DONE ==============================')
        sys.exit("ERROR")


def trim_node(node_id, org_pred, k, appear_dict: AppearDict):
    node_dict = appear_dict.node_dict[node_id]
    roots = deepcopy(node_dict.roots)
    # logger.info(f"Trimming ndoe {node_id} with roots {roots}")
    if node_id in roots:
        roots.remove(node_id)
        k = k - 1
    block_dict = {}
    for root in roots:
        if root not in appear_dict.trim_info.keys():
            appear_dict.trim_info['# subgraphs trimmed'] += 1
            appear_dict.trim_info['trimmed subgraphs'].append(root)
            appear_dict.trim_info[root] = {
                '# nodes org': 0,
                '# edges org': 0,
                '# nodes trimmed': 0,
                '# edges trimmed': 0,
                'trimmed nodes': [],
                'rank trimmed': []
            }

        if node_id not in appear_dict.trim_info[root]['trimmed nodes']:
            appear_dict.trim_info[root]['trimmed nodes'].append(node_id)
            appear_dict.trim_info[root]['# nodes trimmed'] += 1
        root_index = appear_dict.roots_index[root]
        blocks = deepcopy(appear_dict.subgraphs[root])
        ranks = node_dict.root_dict[root]['rank']
        for rank in ranks:
            appear_dict.trim_info[root]['rank trimmed'].append(rank)
            if rank == 0:
                block = blocks[rank]
                src_node = block.srcdata[dgl.NID]
                dst_node = block.dstdata[dgl.NID]
                src_edge, dst_edge = block.edges()
                src_edge = torch.index_select(src_node, 0, src_edge)
                dst_edge = torch.index_select(dst_node, 0, dst_edge)
                num_edge_before = len(src_edge)
                indices = get_index_bynot_value(a=src_edge, val=node_id)
                src_edge = torch.index_select(src_edge, 0, indices)
                dst_edge = torch.index_select(dst_edge, 0, indices)
                num_edge_after = len(src_edge)
                appear_dict.trim_info[root]['# edges trimmed'] += num_edge_before - num_edge_after
                # print(f"Node {node_id} at root {root} and rank {rank} - After:", src_edge, dst_edge)
                g = dgl.graph((src_edge, dst_edge))
                blk = to_block(g, dst_nodes=dst_node, include_dst_in_src=False)
                blocks[rank] = blk
            elif rank > 0 & rank < appear_dict.num_layer:
                block = blocks[rank]
                src_node = block.srcdata[dgl.NID]
                dst_node = block.dstdata[dgl.NID]
                src_edge, dst_edge = block.edges()
                src_edge = torch.index_select(src_node, 0, src_edge)
                dst_edge = torch.index_select(dst_node, 0, dst_edge)
                num_edge_before = len(src_edge)
                indices = get_index_bynot_value(a=src_edge, val=node_id)
                src_edge = torch.index_select(src_edge, 0, indices)
                dst_edge = torch.index_select(dst_edge, 0, indices)
                num_edge_after = len(src_edge)
                appear_dict.trim_info[root]['# edges trimmed'] += num_edge_before - num_edge_after
                # print(f"Node {node_id} at root {root} and rank {rank} - After:", src_edge, dst_edge)
                g = dgl.graph((src_edge, dst_edge))
                blocks[rank] = to_block(g, dst_nodes=dst_node, include_dst_in_src=False)
                block = blocks[rank - 1]
                src_node = block.srcdata[dgl.NID]
                dst_node = block.dstdata[dgl.NID]
                src_edge, dst_edge = block.edges()
                src_edge = torch.index_select(src_node, 0, src_edge)
                dst_edge = torch.index_select(dst_node, 0, dst_edge)
                num_edge_before = len(src_edge)
                # print(f"Node {node_id} at root {root} and rank {rank-1} - Before:", src_edge, dst_edge)
                indices = get_index_bynot_value(a=dst_edge, val=node_id)
                src_edge = torch.index_select(src_edge, 0, indices)
                dst_edge = torch.index_select(dst_edge, 0, indices)
                num_edge_after = len(src_edge)
                appear_dict.trim_info[root]['# edges trimmed'] += num_edge_before - num_edge_after
                # print(f"Node {node_id} at root {root} and rank {rank-1} - After:", src_edge, dst_edge)
                g = dgl.graph((src_edge, dst_edge))
                blocks[rank - 1] = to_block(g, dst_nodes=dst_node, include_dst_in_src=False)
            else:
                pass

        new_blocks = []
        # dst_n = torch.Tensor([root]).int()
        dst_n = [root]
        for i in reversed(range(appear_dict.num_layer)):
            src_node = blocks[i].srcdata[dgl.NID]
            dst_node = blocks[i].dstdata[dgl.NID]
            src_edge, dst_edge = blocks[i].edges()
            src_node_new = torch.index_select(src_node, 0, src_edge)
            dst_node_new = torch.index_select(dst_node, 0, dst_edge)
            src_node_new = torch.cat([src_node_new, dst_node_new], dim=0)
            dst_node_new = torch.cat([dst_node_new, dst_node_new], dim=0)
            g = dgl.graph((src_node_new, dst_node_new), num_nodes=len(appear_dict.graph.nodes()))
            g.ndata['feat'] = appear_dict.graph.ndata['feat']
            g.ndata['label'] = appear_dict.graph.ndata['label']
            blk = to_block(g=g, dst_nodes=dst_n, include_dst_in_src=True)
            dst_n = blk.srcdata[dgl.NID]
            new_blocks.insert(0, blk)

        inputs = new_blocks[0].srcdata['feat']
        predictions = appear_dict.model(new_blocks, inputs)
        val = torch.norm(predictions - org_pred[root_index, :], p=1) / (
                torch.norm(predictions, p=1) + torch.norm(org_pred[root_index, :], p=1))
        block_dict[root] = {
            'block': new_blocks,
            'val': val.item()
        }
    temp_ls = []
    for key, val in block_dict.items():
        # temp_ls = sorted(list(block_dict.items()), key=lambda x: x[1]['val'], reverse=True)
        temp_ls.append((key, val['val']))
    temp_ls = sorted(temp_ls, key=lambda x: x[1], reverse=True)
    # print(f"Val: {temp_ls}, len {len(temp_ls)}, replace {temp_ls[num_appear-k:]}, len replace {len(temp_ls[num_appear-k:])}")
    for key, value in temp_ls[k:]:
        appear_dict.subgraphs[key] = block_dict[key]['block']
        appear_dict.node_dict[node_id].roots.remove(key)
        appear_dict.node_dict[node_id].num_tree -= 1
        del appear_dict.node_dict[node_id].root_dict[key]
    return 1