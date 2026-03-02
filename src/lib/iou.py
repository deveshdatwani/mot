import numpy as np

def iou(target, bboxes):   
    ax1, ay1, ax2, ay2 = target[0, :]
    bx1 = bboxes[:,0]
    by1 = bboxes[:,1]
    bx2 = bboxes[:,2]
    by2 = bboxes[:,3]
    cx1 = np.maximum(ax1, bx1)
    cy1 = np.maximum(ay1, by1)
    cx2 = np.minimum(ax2, bx2)
    cy2 = np.minimum(ay2, by2)
    w = np.maximum(0, cx2 - cx1)
    h = np.maximum(0, cy2 - cy1)
    intersection = h * w
    union = ((ax2 - ax1) * (ay2 - ay1)) + ((bx2 - bx1) * (by2 - by1)) - (intersection)
    return intersection / union