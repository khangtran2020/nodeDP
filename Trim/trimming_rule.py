import numpy as np
from copy import deepcopy
def random_trimming(roots, k, current_node, appear_dict=None, model=None, graph=None):
    r = deepcopy(roots)
    if current_node in r:
        r.remove(current_node)
    r = np.array(r)
    list_of_root = np.random.choice(a=r, size=k, replace=False).astype(int)
    return list_of_root.tolist()


def adhoc_trimming_rank(roots, k, current_node, appear_dict=None, model=None, graph=None):
    r = deepcopy(roots)
    if current_node in r:
        r.remove(current_node)
    rank_at_r = []
    for n in r:
        rank_at_r.append(np.sum(np.array(appear_dict.node_dict[current_node].root_dict[n]['rank']) + 1))
    rout_rank = sorted(list(zip(r, rank_at_r)), key=lambda x: x[1])
    list_of_root = [x[0] for x in rout_rank]
    return list_of_root