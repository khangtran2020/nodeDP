import numpy as np

def random_trimming(roots, k, current_node):
    if current_node in roots:
        roots.remove(current_node)
    roots = np.array(roots)
    list_of_root = np.random.choice(a=roots, size=k, replace=False).astype(int)
    return list_of_root.tolist()