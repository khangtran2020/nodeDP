import dgl
import torch
from Utils.utils import get_index_by_value, get_index_bynot_value
from Trim.trimming_rule import random_trimming, adhoc_trimming_rank
from dgl.dataloading import to_block


class Node(object):

    def __init__(self, node_id, num_layer):
        self.id = node_id
        self.num_tree = 0
        self.num_layer = num_layer
        self.roots = []
        self.root_dict = {}

    def add_sub_graph(self, root, rank, ans=None, child=None):
        if root in self.roots:
            if rank in self.root_dict[root]['rank']:
                if ans is not None:
                    self.root_dict[root][rank]['ans'] = ans
                    self.root_dict[root][rank]['out_deg'] = len(ans)
                if child is not None:
                    self.root_dict[root][rank]['child'] = child
                    self.root_dict[root][rank]['in_deg'] = len(child)
            else:
                self.root_dict[root]['num_appear'] += 1
                self.root_dict[root]['rank'].append(rank)
                self.root_dict[root][rank] = {}
                if ans is not None:
                    self.root_dict[root][rank]['ans'] = ans
                    self.root_dict[root][rank]['out_deg'] = len(ans)
                if child is not None:
                    self.root_dict[root][rank]['child'] = child
                    self.root_dict[root][rank]['in_deg'] = len(child)
        else:
            self.num_tree += 1
            self.roots.append(root)
            self.root_dict[root] = {
                'num_appear': 1,
                'rank': [rank],
                rank: {}
            }
            if ans is not None:
                self.root_dict[root][rank]['ans'] = ans
                self.root_dict[root][rank]['out_deg'] = len(ans)
            if child is not None:
                self.root_dict[root][rank]['child'] = child
                self.root_dict[root][rank]['in_deg'] = len(child)

    def get_num_tree(self):
        return self.num_tree

    def get_ans(self, root, rank):
        if root not in self.roots:
            print(f"Func get_ans: Node {self.id} is not in subgraph rooted at {root}")
            return
        if rank not in self.root_dict[root].keys():
            print(f"Func get_ans: Node {self.id} is not in rank {rank} of root {root}")
            return
        if rank == self.num_layer - 1:
            print(f"Func get_ans: Node {self.id} is in rank {rank} and don't have anscestors")
            return
        return self.root_dict[root][rank]['ans']

    def get_out_deg(self, root, rank):
        if root not in self.roots:
            print(f"Func get_out_deg: Node {self.id} is not in subgraph rooted at {root}")
            return
        if rank not in self.root_dict[root].keys():
            print(f"Func get_out_deg: Node {self.id} is not in rank {rank} of root {root}")
            return
        if rank == self.num_layer - 1:
            print(f"Func get_out_deg: Node {self.id} is in rank {rank} and don't have anscestors")
            return
        return self.root_dict[root][rank]['out_deg']

    def get_child(self, root, rank):
        if root not in self.roots:
            print(f"Func get_child: Node {self.id} is not in subgraph rooted at {root}")
            return
        if rank not in self.root_dict[root].keys():
            print(f"Func get_child: Node {self.id} is not in rank {rank} of root {root}")
            return
        if rank == 0:
            print(f"Func get_child: Node {self.id} is in rank {rank} and don't have childs")
            return
        return self.root_dict[root][rank]['child']

    def get_in_deg(self, root, rank):
        if root not in self.roots:
            print(f"Func get_in_deg: Node {self.id} is not in subgraph rooted at {root}")
            return
        if rank not in self.root_dict[root].keys():
            print(f"Func get_in_deg: Node {self.id} is not in rank {rank} of root {root}")
            return
        if rank == 0:
            print(f"Func get_in_deg: Node {self.id} is in rank {rank} and don't have childs")
            return
        return self.root_dict[root][rank]['in_deg']

    def del_child(self, node_id, root, rank):
        if root not in self.roots:
            print(f"Func del_child: Node {self.id} is not in subgraph rooted at {root}")
            return
        if rank not in self.root_dict[root].keys():
            print(f"Func del_child: Node {self.id} is not in rank {rank} of root {root}")
            return
        if rank == 0:
            print(f"Func del_child: Node {self.id} is in rank {rank} and don't have childs")
            return
        if node_id not in self.root_dict[root][rank]['child']:
            print(f'Func del_child: Node {node_id} is not the child of node {self.id} in root {root}')
            return
        self.root_dict[root][rank]['child'].remove(node_id)
        self.root_dict[root][rank]['in_deg'] -= 1

    def del_ans(self, node_id, root, rank):
        if root not in self.roots:
            print(f"Func del_ans: Node {self.id} is not in subgraph rooted at {root}")
            return
        if rank not in self.root_dict[root].keys():
            print(f"Func del_ans: Node {self.id} is not in rank {rank} of root {root}")
            return
        if rank == self.num_layer - 1:
            print(f"Func del_ans: Node {self.id} is in rank {rank} and don't have ancestors")
            return
        if node_id not in self.root_dict[root][rank]['ans']:
            print(f'Func del_ans: Node {node_id} is not the ancestor of node {self.id} in root {root}')
            return
        self.root_dict[root][rank]['ans'].remove(node_id)
        self.root_dict[root][rank]['out_deg'] -= 1


class AppearDict(object):

    def __init__(self, roots, subgraphs, trimming_rule=None, k=2):
        self.roots = roots
        self.subgraphs = subgraphs
        self.node_dict = self.build_node_dict(roots, subgraphs)
        self.num_appear = self.get_num_tree()
        self.trimming_rule = trimming_rule
        if trimming_rule == 'random':
            self.trimming_func = random_trimming
        elif trimming_rule == 'adhoc':
            self.trimming_func = adhoc_trimming_rank
        self.k = k

    def get_num_tree(self):
        keys = []
        vals = []
        for n in self.node_dict.keys():
            keys.append(n)
            vals.append(self.node_dict[n].num_tree)
        temp = list(zip(keys, vals))
        return sorted(temp, key=lambda x: x[1], reverse=True)

    def build_node_dict(self, roots, subgraphs):
        results = {}
        for i, node in enumerate(roots):
            blocks = subgraphs[node]
            num_layer = len(blocks) + 1
            for j, block in enumerate(blocks):
                ans_rank = j + 1
                child_rank = j
                src_node = block.srcdata[dgl.NID]
                dst_node = block.dstdata[dgl.NID]
                src_edge, dst_edge = block.edges()
                src_node_new = torch.index_select(src_node, 0, src_edge)
                dst_node_new = torch.index_select(dst_node, 0, dst_edge)
                for n in dst_node.tolist():
                    if n not in results.keys():
                        results[n] = Node(node_id=n, num_layer=num_layer)
                    indices = get_index_by_value(a=dst_node_new, val=n)
                    child = torch.index_select(src_node_new, 0, indices).unique().tolist()
                    results[n].add_sub_graph(root=node, rank=ans_rank, ans=None, child=child)
                for n in src_node.tolist():
                    if n not in results.keys():
                        results[n] = Node(node_id=n, num_layer=num_layer)
                    indices = get_index_by_value(a=src_node_new, val=n)
                    ancestor = torch.index_select(dst_node_new, 0, indices).unique().tolist()
                    results[n].add_sub_graph(root=node, rank=child_rank, ans=ancestor, child=None)
        return results

    def getkeys(self):
        return self.node_dict.keys()

    def get_list_trimming_root(self, node_id):
        if self.trimming_rule == 'random':
            return self.trimming_func(roots=self.node_dict[node_id].roots, k=self.node_dict[node_id].num_tree - self.k,
                                      current_node=node_id, appear_dict=None, model=None, graph=None)
        elif self.trimming_rule == 'adhoc':
            return self.trimming_func(roots=self.node_dict[node_id].roots, k=self.node_dict[node_id].num_tree - self.k,
                                      current_node=node_id, appear_dict=self, model=None, graph=None)

    def trim(self):
        node_appear = self.get_num_tree()
        highest_appeared_node = node_appear[0][0]
        val = node_appear[0][1]
        while self.node_dict[highest_appeared_node].num_tree > self.k:
            self.trim_node(node_id=highest_appeared_node)
            node_appear = self.get_num_tree()
            highest_appeared_node = node_appear[0][0]
            val = node_appear[0][1]
        self.num_appear = self.get_num_tree()

    def trim_node(self, node_id):
        list_of_root = self.get_list_trimming_root(node_id=node_id)
        for root in list_of_root:
            blocks = self.subgraphs[root]
            ranks = self.node_dict[node_id].root_dict[root]['rank']
            queue = []
            for r in ranks:
                queue.append((node_id, r))
            while (len(queue) > 0):
                n, r = queue[0]
                queue.pop(0)
                # if r not in self.node_dict[node_id].roots:
                #     continue
                if r == 0:
                    blocks[r], queue = self.remove_node_from_root_at_rank(node_id=n, root=root, rank=r,
                                                                          blocks=[blocks[r]],
                                                                          queue=queue)
                elif (r < len(blocks)) and (r > 0):
                    blocks[r], blocks[r - 1], queue = self.remove_node_from_root_at_rank(node_id=n, root=root,
                                                                                         rank=r,
                                                                                         blocks=[blocks[r],
                                                                                                 blocks[r - 1]],
                                                                                         queue=queue)
                else:
                    print(f"Node {node_id} is the root, can not be remove")
                    return
                self.subgraphs[root] = blocks

    def remove_node_from_root_at_rank(self, node_id, root, rank, blocks, queue):
        if rank == 0:
            block = blocks[0]
            src_node = block.srcdata[dgl.NID]
            dst_node = block.dstdata[dgl.NID]
            src_edge, dst_edge = block.edges()
            src_node_new = torch.index_select(src_node, 0, src_edge)
            dst_node_new = torch.index_select(dst_node, 0, dst_edge)
            indices = get_index_bynot_value(src_node_new, val=node_id)
            src_new = torch.index_select(input=src_node_new, dim=0, index=indices)
            dst_new = torch.index_select(input=dst_node_new, dim=0, index=indices)
            g = dgl.graph((src_new, dst_new))
            # update appear_dict
            if self.node_dict[node_id].root_dict[root]['num_appear'] > 1:
                self.node_dict[node_id].root_dict[root]['num_appear'] -= 1
                self.node_dict[node_id].root_dict[root]['rank'].remove(rank)
                ancestor = self.node_dict[node_id].get_ans(root=root, rank=rank)
                for n in ancestor:
                    self.node_dict[n].del_child(node_id=node_id, root=root, rank=rank + 1)
                del self.node_dict[node_id].root_dict[root][rank]
            else:
                ancestor = self.node_dict[node_id].get_ans(root=root, rank=rank)
                for n in ancestor:
                    self.node_dict[n].del_child(node_id=node_id, root=root, rank=rank + 1)
                self.node_dict[node_id].num_tree -= 1
                if node_id != root:
                    self.node_dict[node_id].roots.remove(root)
                    del self.node_dict[node_id].root_dict[root]
            return to_block(g, dst_nodes=dst_node, include_dst_in_src=False), queue
        else:
            block_r = blocks[0]
            block_r1 = blocks[1]
            src_node = block_r.srcdata[dgl.NID]
            dst_node = block_r.dstdata[dgl.NID]
            src_edge, dst_edge = block_r.edges()
            src_node_new = torch.index_select(src_node, 0, src_edge)
            dst_node_new = torch.index_select(dst_node, 0, dst_edge)
            indices = get_index_bynot_value(src_node_new, val=node_id)
            src_new = torch.index_select(input=src_node_new, dim=0, index=indices)
            dst_new = torch.index_select(input=dst_node_new, dim=0, index=indices)
            g = dgl.graph((src_new, dst_new))
            block_r = to_block(g, dst_nodes=dst_node, include_dst_in_src=False)

            src_node = block_r1.srcdata[dgl.NID]
            dst_node = block_r1.dstdata[dgl.NID]
            src_edge, dst_edge = block_r1.edges()
            src_node_new = torch.index_select(src_node, 0, src_edge)
            dst_node_new = torch.index_select(dst_node, 0, dst_edge)
            indices = get_index_bynot_value(dst_node_new, val=node_id)
            src_new = torch.index_select(input=src_node_new, dim=0, index=indices)
            dst_new = torch.index_select(input=dst_node_new, dim=0, index=indices)
            index_not_node_id = get_index_bynot_value(dst_node, val=node_id)
            dst_node = torch.index_select(input=dst_node, dim=0, index=index_not_node_id)
            g = dgl.graph((src_new, dst_new))
            block_r1 = to_block(g, dst_nodes=dst_node, include_dst_in_src=False)


            ancestor = self.node_dict[node_id].get_ans(root=root, rank=rank)
            for n in ancestor:
                self.node_dict[n].del_child(node_id=node_id, root=root, rank=rank + 1)
            childs = self.node_dict[node_id].get_child(root=root, rank=rank)
            for n in childs:
                self.node_dict[n].del_ans(node_id=node_id, root=root, rank=rank - 1)
                if (self.node_dict[n].root_dict[root][rank - 1]['out_deg']) <= 0 and (n != root):
                    queue.append((n, rank - 1))

            if self.node_dict[node_id].root_dict[root]['num_appear'] > 1:
                self.node_dict[node_id].root_dict[root]['num_appear'] -= 1
                self.node_dict[node_id].root_dict[root]['rank'].remove(rank)
                del self.node_dict[node_id].root_dict[root][rank]
            else:
                self.node_dict[node_id].num_tree -= 1
                if (node_id != root):
                    self.node_dict[node_id].roots.remove(root)
                    del self.node_dict[node_id].root_dict[root]
            return (block_r, block_r1, queue)

    def build_blocks(self, root, graph):
        blocks = self.subgraphs[root]
        new_blocks = []
        dst_n = root
        for i, block in enumerate(reversed(blocks)):
            src_node = block.srcdata[dgl.NID]
            dst_node = block.dstdata[dgl.NID]
            src_edge, dst_edge = block.edges()
            src_node_new = torch.index_select(src_node, 0, src_edge)
            dst_node_new = torch.index_select(dst_node, 0, dst_edge)
            src_node_new = torch.cat([src_node_new, dst_node_new], dim=0)
            dst_node_new = torch.cat([dst_node_new, dst_node_new], dim=0)
            g = dgl.graph((src_node_new, dst_node_new), num_nodes=len(graph.nodes()))
            g.ndata['feat'] = graph.ndata['feat']
            g.ndata['label'] = graph.ndata['label']
            blk = to_block(g=g, dst_nodes=dst_n, include_dst_in_src=True)
            dst_n = blk.srcdata[dgl.NID]
            new_blocks.insert(0, blk)
        return new_blocks

    def print_nodes(self):
        for key, val in self.get_num_tree():
            print(key, '\t', self.node_dict[key].num_tree, self.node_dict[key].roots, self.node_dict[key].root_dict)

    def print_root(self, roots):
        for root in roots:
            print(root, self.node_dict[root].root_dict)