import dgl
import torch
import multiprocessing

from Trim.trimming_rule import random_trimming
from dgl.dataloading import to_block


def get_node_counts(dst_node, subtree):
    appear_dict = {}
    subtree_node_dict = {}
    for i, node in enumerate(dst_node):
        blocks = subtree[node.item()]
        did_appear = []
        did_appear.append(node)
        if node not in appear_dict.keys():
            appear_dict[node] = {
                'num_tree': 1,
                'root': [node],
                node: {
                    'num_appear': 1,
                    'block': [-1]
                }
            }
        else:
            appear_dict[node]['num_tree'] += 1
            appear_dict[node]['root'].append(node)
            appear_dict[node][node] = {
                'num_appear': 1,
                'block': [-1]
            }
        subtree_node_dict[node] = {}
        for j, block in enumerate(reversed(blocks)):
            subtree_node_dict[node][f'block_{j}'] = []
            src_node = block.srcdata[dgl.NID]
            unique_src_node = src_node[block.num_dst_nodes():]
            # print(unique_src_node)
            for n in unique_src_node.tolist():
                subtree_node_dict[node][f'block_{j}'].append(n)
                if n not in appear_dict.keys():
                    did_appear.append(n)
                    appear_dict[n] = {
                        'num_tree': 1,
                        'root': [node],
                        node: {
                            'num_appear': 1,
                            'block': [len(blocks) - j - 1]
                        }
                    }
                elif (n in appear_dict.keys()) and (n not in did_appear):
                    did_appear.append(n)
                    appear_dict[n]['num_tree'] += 1
                    appear_dict[n]['root'].append(node)
                    appear_dict[n][node] = {
                        'num_appear': 1,
                        'block': [len(blocks) - j - 1]
                    }
                elif (n in appear_dict.keys()) and (n in did_appear):
                    appear_dict[n][node]['num_appear'] += 1
                    appear_dict[n][node]['block'].append(len(blocks) - j - 1)

    node_appear = []
    for key, val in appear_dict.items():
        node_appear.append((key, val['num_tree']))
    return appear_dict, subtree_node_dict, node_appear


def sort_by_num_tree(appear_dict):
    node_appear = [(x, val['num_tree']) for x, val in appear_dict.items()]
    sorted_by_num_tre = sorted(node_appear, key=lambda x: x[1], reverse=True)
    return sorted_by_num_tre


def trim_subgraph(highest_appeared_node, root, sub_graph, appear_dict, subtree_node_dict):
    blocks = sub_graph[root]
    # print('Before', blocks)
    num_layer = len(blocks)
    appear_in = appear_dict[highest_appeared_node][root]['block']
    num_appear = appear_dict[highest_appeared_node][root]['num_appear']
    # print(root, appear_in)
    for i in appear_in:
        current_block = i
        queue = [(highest_appeared_node, current_block, num_appear)]
        remove_from_block(queue, blocks, appear_dict, root, subtree_node_dict)
    # print('After', blocks)


def remove_from_block(queue, blocks, appear_dict, root, subtree_node_dict):
    while (len(queue) > 0):
        # print('Queue:', queue)
        current_node, current_block, num_appear = queue[0]
        queue.pop(0)
        if current_block > 0:
            blocks[current_block] = remove_src_block(current_node, blocks[current_block], appear_dict, root,
                                                     subtree_node_dict)
            queue = construct_queue_next(queue=queue, block=blocks[current_block - 1], current_node=current_node,
                                         block_id=current_block - 1, appear_dict=appear_dict, root=root)
        else:
            blocks[current_block] = remove_src_block(current_node, blocks[current_block], appear_dict, root,
                                                     subtree_node_dict)



def construct_queue_next(queue, block, current_node, block_id, appear_dict, root):
    src_edge, dst_edge = block.edges()
    src_nodes = block.srcdata[dgl.NID][len(block.dstdata[dgl.NID]):]
    if len(src_nodes) == 0:
        return queue
    dst_nodes = block.dstdata[dgl.NID]
    id_x = (block.dstdata[dgl.NID] == current_node).nonzero(as_tuple=True)[0]
    if len(id_x) == 0:
        return queue
    id_c = (dst_edge == id_x).nonzero(as_tuple=True)[0]
    if len(id_c > 0):
        sub_child = torch.clip(src_edge[id_c]-len(block.dstdata[dgl.NID]), min=int(0))
        sub_child = torch.index_select(src_nodes, 0, sub_child).unique().tolist()
        for n in sub_child:
            if root in appear_dict[n]['root']:
                num_appear = appear_dict[n][root]['num_appear']
                queue.append((n, block_id, num_appear))
        return queue
    else:
        return queue


def remove_src_block(current_node, block, appear_dict, root, subtree_node_dict):
    src_edge, dst_edge = block.edges()
    if current_node == root:
        return block
    nodes = block.srcdata[dgl.NID]
    idx = (nodes == current_node).nonzero(as_tuple=True)[0]
    # print(idx, current_node, nodes)
    ide = (src_edge != idx.item()).nonzero(as_tuple=True)[0]
    src_new = src_edge[ide]
    dst_new = dst_edge[ide]
    src_node_new = torch.index_select(nodes, 0, src_new)
    dst_node_new = torch.index_select(nodes, 0, dst_new)
    g = dgl.DGLGraph((src_node_new, dst_node_new))
    block = to_block(g=g, dst_nodes=dst_node_new.unique())
    if appear_dict[current_node][root]['num_appear'] > 1:
        appear_dict[current_node][root]['num_appear'] -= 1
    else:
        appear_dict[current_node]['num_tree'] -= 1
        appear_dict[current_node]['root'].remove(root)
        del appear_dict[current_node][root]
    return block


def trim(appear_dict, sub_graph, num_worker, sampling_rule, k, subtree_node_dict):
    # choose sampling rule
    sample = None
    if sampling_rule == 'random':
        sample = random_trimming
    # sort appear_dict
    node_appear = sort_by_num_tree(appear_dict=appear_dict)
    highest_appeared_node = node_appear[0][0]
    print('Current highest node:', highest_appeared_node)
    while appear_dict[highest_appeared_node]['num_tree'] > k:
        # build the queue
        # sampling -> return list of root to be trim
        list_of_root = sample(roots=appear_dict[highest_appeared_node]['root'],
                              k=appear_dict[highest_appeared_node]['num_tree'] - k, current_node=highest_appeared_node)
        for r in list_of_root:
            trim_subgraph(highest_appeared_node=highest_appeared_node, root=r, sub_graph=sub_graph,
                          appear_dict=appear_dict, subtree_node_dict=subtree_node_dict)
        # multiporcessing
        # num_process = num_worker if len(list_of_root) > num_worker else len(list_of_root)
        # pool = multiprocessing.Pool(num_process)
        # processes = [pool.apply(trim_subgraph, args=(highest_appeared_node, n, sub_graph, appear_dict)) for n in
        #              list_of_root]
        # for p in processes:
        #     p.get()
        node_appear = sort_by_num_tree(appear_dict=appear_dict)
        highest_appeared_node = node_appear[0][0]
    return appear_dict, node_appear
