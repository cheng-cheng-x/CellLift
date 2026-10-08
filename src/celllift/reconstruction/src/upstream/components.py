from celllift.runtime import resource_path as _public_resource
from celllift.runtime import resource_path as _resource_path
import numpy as np

def _union_find_components(count: int, edges: np.ndarray) -> np.ndarray:
    parent = np.arange(count, dtype=np.int64)

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = int(parent[root])
        while parent[value] != value:
            nxt = int(parent[value])
            parent[value] = root
            value = nxt
        return root
    for left, right in edges.T if edges.size else ():
        a, b = (find(int(left)), find(int(right)))
        if a != b:
            if a > b:
                a, b = (b, a)
            parent[b] = a
    roots = np.asarray([find(index) for index in range(count)], dtype=np.int64)
    unique = {root: index for index, root in enumerate(np.unique(roots))}
    return np.asarray([unique[int(root)] for root in roots], dtype=np.int64)
