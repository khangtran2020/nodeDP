import numpy as np
from copy import deepcopy
def random_trimming(roots, k, current_node):
    r = deepcopy(roots)
    if current_node in r:
        r.remove(current_node)
    r = np.array(r)
    list_of_root = np.random.choice(a=r, size=k, replace=False).astype(int)
    return list_of_root.tolist()