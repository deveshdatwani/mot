import numpy as np

class Node(object):
    def __init__(self, data, left=None, right=None):
        self.data = data
        self.left = left
        self.right = right

    
def build_kdtree(points, depth=0):
    if len(points) == 0:
        return None
    axis = depth % points.shape[1]
    points = points[points[:, axis].argsort()]
    median  = len(points) // 2
    return Node(points[median],
                build_kdtree(points[:median], depth + 1),
                build_kdtree(points[median + 1:], depth + 1))