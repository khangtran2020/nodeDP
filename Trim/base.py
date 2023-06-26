import dgl
import sys
import torch
import numpy as np
from rich import print as rprint
from rich.pretty import pretty_repr
from Utils.utils import get_index_by_value, get_index_bynot_value
from Trim.trimming_rule import random_trimming, adhoc_trimming_rank, impact_aware_trimming
from dgl.dataloading import to_block
from loguru import logger
from copy import deepcopy

# logger.remove()
logger.add(sys.stderr, format="{time} {level} {message}", filter="my_module", level="INFO")

def get_edge_ls(list_):
    src_edge = []
    dst_edge = []
    for item in list_:
        key = item[0]
        vals = item[1]
        for val in vals:
            src_edge.append(key)
            dst_edge.append(val)
    return src_edge, dst_edge


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

    def __init__(self, roots, subgraphs, id=0, model=None, graph=None, rule=None, k=2, num_layer=2, debug=False):
        self.roots = roots
        self.id = id
        self.subgraphs = subgraphs
        if debug:
            self.trim_info = {
                'num_node_org': 0,
                'num_edge_org': 0,
                'num_subgraph': len(roots),
            }
        self.node_dict, self.root_dict = self.build_node_dict(roots, subgraphs)
        self.num_appear = self.get_num_tree()
        self.trimming_rule = rule
        if rule == 'random':
            self.trimming_func = random_trimming
        elif rule == 'adhoc':
            self.trimming_func = adhoc_trimming_rank
        elif rule == 'impact':
            self.trimming_func = impact_aware_trimming
        self.k = k
        self.model = model
        self.graph = graph
        self.num_layer = num_layer
        self.debug = debug
        if self.debug:
            self.deleted_node = {}
            for root in roots:
                self.deleted_node[root] = []
            self.copy = AppearDict(roots=roots, subgraphs=subgraphs, id=1, model=model, graph=graph, rule=rule, k=k,
                                   num_layer=num_layer, debug=False)

    def get_num_tree(self):
        ls = []
        for node, val in self.node_dict.items():
            ls.append((node, val.num_tree))
        return sorted(ls, key=lambda x: x[1], reverse=True)

    def build_node_dict(self, roots, subgraphs):
        results = {}
        root_dict_ = {}
        for root in roots:
            root_dict = {
                '# nodes org': 0,
                '# edges org': 0,
            }
            blocks = subgraphs[root]
            for i, block in enumerate(blocks):
                ans_rank = i + 1
                child_rank = i
                if f'rank_{child_rank}' not in root_dict.keys():
                    root_dict[f'rank_{child_rank}'] = {}
                if f'rank_{ans_rank}' not in root_dict.keys():
                    root_dict[f'rank_{ans_rank}'] = {}
                src_node = block.srcdata[dgl.NID]
                dst_node = block.dstdata[dgl.NID]
                src_edge, dst_edge = block.edges()
                for j, n in enumerate(dst_node.tolist()):

                    if n not in results.keys():
                        results[n] = Node(node_id=n)

                    if n not in root_dict[f'rank_{ans_rank}'].keys():
                        root_dict[f'rank_{ans_rank}'][n] = {
                            'child': [],
                            'ans': []
                        }
                        root_dict['# nodes org'] += 1

                    indices = get_index_by_value(a=dst_edge, val=j)
                    child_idx = torch.index_select(src_edge, 0, indices).unique()
                    child = torch.index_select(src_node, 0, child_idx).tolist()

                    results[n].add_sub_graph(root=root, rank=ans_rank)
                    root_dict[f'rank_{ans_rank}'][n]['child'] += child
                    root_dict['# edges org'] += len(child)
                for j, n in enumerate(src_node.tolist()):

                    if n not in results.keys():
                        results[n] = Node(node_id=n)

                    if n not in root_dict[f'rank_{child_rank}'].keys():
                        root_dict[f'rank_{child_rank}'][n] = {
                            'child': [],
                            'ans': []
                        }
                        root_dict['# nodes org'] += 1
                    indices = get_index_by_value(a=src_edge, val=j)
                    ans_idx = torch.index_select(dst_edge, 0, indices).unique()
                    ancestor = torch.index_select(dst_node, 0, ans_idx).tolist()
                    results[n].add_sub_graph(root=root, rank=child_rank)
                    root_dict[f'rank_{child_rank}'][n]['ans'] += ancestor
                    root_dict['# edges org'] += len(ancestor)
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

    def get_list_trimming_root(self, node_id):
        if self.trimming_rule == 'random':
            return self.trimming_func(roots=self.node_dict[node_id].roots, k=self.node_dict[node_id].num_tree - self.k,
                                      current_node=node_id, appear_dict=None, model=None, graph=None)
        elif self.trimming_rule == 'adhoc':
            return self.trimming_func(roots=self.node_dict[node_id].roots, k=self.node_dict[node_id].num_tree - self.k,
                                      current_node=node_id, appear_dict=self, model=None, graph=None)
        elif self.trimming_rule == 'impact':
            return self.trimming_func(roots=self.node_dict[node_id].roots, k=self.node_dict[node_id].num_tree - self.k,
                                      current_node=node_id, appear_dict=self, model=self.model, graph=self.graph)

    def trim(self):
        # i = 0
        node_appear = self.get_num_tree()
        # rprint(node_appear)
        highest_appeared_node, val = node_appear[0]
        trimmed_root_ls = []
        while val > self.k:
            # print(f"Starting {i} process for node {highest_appeared_node} which appears {self.node_dict[highest_appeared_node].num_tree}")
            self.trim_node(node_id=highest_appeared_node, root_ls=trimmed_root_ls)
            # print(f"{highest_appeared_node} now appears {self.node_dict[highest_appeared_node].num_tree} after process {i}")
            node_appear = self.get_num_tree()
            highest_appeared_node, val = node_appear[0]
            # i += 1
        # for root in trimmed_root_ls:
        # rprint(f"Root {root}: "
        #        f"# node org {self.root_dict[root]['# nodes org']}, # edge org {self.root_dict[root]['# edges org']}, "
        #        f"% of node removed {self.root_dict[root]['# trimmed nodes']/self.root_dict[root]['# nodes org']:.4f}, "
        #        f"% of edge removed {self.root_dict[root]['# trimmed edges']/self.root_dict[root]['# edges org']:.4f}")
        if self.debug:
            self.check_nodes()

    def check_nodes_init(self):
        appear_dict = {}
        for root in self.roots:
            root_dict = self.root_dict[root]
            for i in range(self.num_layer):
                for n in root_dict[f'rank_{i}'].keys():
                    if n not in appear_dict.keys():
                        appear_dict[n] = {
                            'roots': [root],
                            'num_appear': 1
                        }
                    else:
                        if root not in appear_dict[n]['roots']:
                            appear_dict[n]['roots'].append(root)
                            appear_dict[n]['num_appear'] += 1
        for key, val in appear_dict.items():
            rprint(f"Node {key} appears {val['num_appear']} times in roots {val['roots']}")

    def check_nodes(self):
        appear_dict = {}
        for root in self.roots:
            root_dict = self.root_dict[root]
            for i in range(self.num_layer):
                for n in root_dict[f'rank_{i}'].keys():
                    if n not in appear_dict.keys():
                        appear_dict[n] = {
                            'roots': [root],
                            'num_appear': 1
                        }
                    else:
                        if root not in appear_dict[n]['roots']:
                            appear_dict[n]['roots'].append(root)
                            appear_dict[n]['num_appear'] += 1
        temp = sorted(list(appear_dict.items()), key=lambda x: x[1]['num_appear'], reverse=True)
        if temp[0][1]['num_appear'] > self.k:
            logger.error(f"Node {temp[0][0]} appeared {temp[0][1]['num_appear']} times at roots {temp[0][1]['roots']}.")
            for root in temp[0][1]['roots']:
                root_dict = self.build_root_dict(root=root)
                rprint(f'Dict of root {root}: \n{pretty_repr(root_dict)}')
            sys.exit("ERROR: you stupid mother fucker!")

    def trim_node(self, node_id, root_ls):
        list_of_root = self.get_list_trimming_root(node_id=node_id)
        for root in list_of_root:
            if root not in root_ls: root_ls.append(root)
            if '# trimmed nodes' not in self.root_dict[root].keys(): self.root_dict[root]['# trimmed nodes'] = 0
            if '# trimmed edges' not in self.root_dict[root].keys(): self.root_dict[root]['# trimmed edges'] = 0
            ranks = self.node_dict[node_id].root_dict[root]['rank']
            queue = []
            for r in ranks: queue.append((node_id, r))
            while (len(queue) > 0):
                n, r = queue[0]
                # logger.info(f'\tTriming node {n} at root {root} rank {r}:')
                queue.pop(0)
                if self.debug == True:
                    self.deleted_node[root].append((n, r))
                if r == 0:
                    res, queue = self.remove_node_from_root_at_rank(node_id=n, root=root, rank=r, queue=queue)
                elif (r < self.num_layer) and (r > 0):
                    res, queue = self.remove_node_from_root_at_rank(node_id=n, root=root, rank=r, queue=queue)
                else:
                    print(f"Node {node_id} is the root, can not be remove")
                    return
                if res == -1:
                    self.handle_error(root=root, node=n, rank=r)
                # logger.info(f'\tTrimmed node {n} at root {root} rank {r} with results {res}!')
        return 1

    def remove_node_from_root_at_rank(self, node_id, root, rank, queue):
        if node_id == root:
            logger.error(f"Queue error: node {node_id} is root of root {root}")
            return (-1, queue)
        self.root_dict[root]['# trimmed nodes'] += 1
        # update root dict
        if rank == 0:
            try:
                ans = self.root_dict[root][f'rank_{rank}'][node_id]['ans']
                del self.root_dict[root][f'rank_{rank}'][node_id]
            except Exception as e:
                logger.error(e)
                self.handle_error(root=root, node=node_id, rank=rank)

            for n in ans:
                if node_id not in self.root_dict[root][f'rank_{rank + 1}'][n]['child']:
                    logger.error(f"Root Dict: Error at del child {node_id} of node {n}, in root {root}, at rank {rank}")
                    return (-1, queue)
                self.root_dict[root]['# trimmed edges'] += 1
                self.root_dict[root][f'rank_{rank + 1}'][n]['child'].remove(node_id)
        else:
            # update root dict
            try:
                ans = self.root_dict[root][f'rank_{rank}'][node_id]['ans']
                child = self.root_dict[root][f'rank_{rank}'][node_id]['child']
                del self.root_dict[root][f'rank_{rank}'][node_id]
            except Exception as e:
                logger.error(e)
                print(f'appear dict id {self.id}', pretty_repr(self.root_dict[root][f'rank_{rank}']))
                self.handle_error(root=root, node=node_id, rank=rank)

            # del ans/child
            for n in ans:
                if node_id not in self.root_dict[root][f'rank_{rank + 1}'][n]['child']:
                    logger.error(f"Root Dict: Error at del child {node_id} of node {n}, in root {root}, at rank {rank}")
                    return (-1, queue)
                self.root_dict[root]['# trimmed edges'] += 1
                self.root_dict[root][f'rank_{rank + 1}'][n]['child'].remove(node_id)
            for n in child:
                if node_id not in self.root_dict[root][f'rank_{rank - 1}'][n]['ans']:
                    logger.error(f"Root Dict: Error at del ans {node_id} of node {n}, in root {root}, at rank {rank}")
                    return (-1, queue)
                self.root_dict[root]['# trimmed edges'] += 1
                self.root_dict[root][f'rank_{rank - 1}'][n]['ans'].remove(node_id)
                if (len(self.root_dict[root][f'rank_{rank - 1}'][n]['ans']) > 0) and (n != root):
                    queue.append((n, rank - 1))

        # update node dict
        if self.node_dict[node_id].root_dict[root]['num_appear'] > 1:
            self.node_dict[node_id].root_dict[root]['num_appear'] -= 1
            if rank not in self.node_dict[node_id].root_dict[root]['rank']:
                logger.error(f"Node Dict: Error at del rank {rank} of node {node_id}, in root {root}")
                return (-1, queue)
            self.node_dict[node_id].root_dict[root]['rank'].remove(rank)
        else:
            self.node_dict[node_id].num_tree -= 1
            self.node_dict[node_id].root_dict[root]['num_appear'] -= 1
            if root not in self.node_dict[node_id].roots:
                logger.error(f"Node Dict: Error at del root {root} of node {node_id}")
                return (-1, queue)
            self.node_dict[node_id].roots.remove(root)
            del self.node_dict[node_id].root_dict[root]

        return (1, queue)

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

    def print_nodes(self):
        node_dict = {}
        for node in self.node_dict.keys():
            node_dict[node] = self.node_dict[node].root_dict
        rprint('Node dict: \n', pretty_repr(node_dict))

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
