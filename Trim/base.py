import dgl
import numpy as np


def get_node_counts(dst_node, subtree):
    appear_dict = {}
    for i, node in enumerate(dst_node):
        blocks = subtree[node.item()]
        num_layer = len(blocks)
        did_appear = []
        did_appear.append(node)
        appear_dict[node] = {
            'num_tree': 1,
            'root': [node],
            node: {
                'num_appear': 1,
                'rank': [0]
            }
        }
        for i, block in enumerate(reversed(blocks)):
            src_node = block.srcdata[dgl.NID]
            unique_src_node = src_node[block.num_dst_nodes() :]
            # print(unique_src_node)
            for n in unique_src_node.tolist():
                if n not in appear_dict.keys():
                    did_appear.append(n)
                    appear_dict[n] = {
                        'num_tree': 1,
                        'root': [node],
                        node: {
                            'num_appear': 1,
                            'rank': [i+1]
                        }
                    }
                elif (n in appear_dict.keys()) and (n not in did_appear):
                    did_appear.append(n)
                    appear_dict[n]['num_tree'] += 1
                    appear_dict[n]['root'].append(node)
                    appear_dict[n][node] = {
                        'num_appear': 1,
                        'rank': [i+1]
                    }
                elif (n in appear_dict.keys()) and (n in did_appear):
                    appear_dict[n][node]['num_appear'] += 1
                    appear_dict[n][node]['rank'].append(i+1)
    return appear_dict

