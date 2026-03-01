import numpy as np

class Node(object):
    def __init__(self, data):
        self.data = data
        self.left = None
        self.right = None

np.random.seed(1)
x = np.random.randint(size=(123,3), low=0, high=100)

def insert(root, x, depth):
    if root is None:
        return Node(x)
    elif root.data[depth % x.shape[-1]] > x[depth % x.shape[-1]]:
        root.left = insert(root.left, x, depth+1)
    else: 
        root.right = insert(root.right, x, depth+1)
    return root

def build_tree(x):
    main_root = Node(x[0])
    for x_ in x[1:]:
        insert(main_root, x_, 0)
    return main_root

x_root = build_tree(x)