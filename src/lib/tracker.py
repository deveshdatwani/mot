import numpy as np
import cv2


class Tracker(object):
    def __init__(self, max_miss=5):
        self.next_id = 0
        self.tracks = {}
        self.max_miss = max_miss
        self.dt = 1/30
        self.A = np.array([[1, 0, self.dt, 0],
                           [0, 1, 0, self.dt],
                           [0, 0, 1, 0],
                           [0, 0, 0, 1]])
        self.X = np.array([[0], [0], [0], [0]]) 
        self.Q = np.eye(4) * 0.01
        self.R = np.eye(2) * 0.01
        self.P = np.eye(4) * 0.01
    
    def predict(self, dt):
        self.A[0, 2] = dt
        self.A[1, 3] = dt
        self.X = self.A @ self.X
        self.P = (self.A @ (self.P @ self.A.T)) + self.Q
        return self.X[:2].flatten()
    
    def update(self, z):
        z = z.reshape((2, 1))
        S = (self.H @ (self.P @ self.H.T)) + self.R
        K = (self.P @ self.H.T) @ np.linalg.inv(S)
        y = z - (self.H @ self.X)
        self.X = self.X + (K @ y)
        self.P = self.P - ((K @ self.H) @ self.P)
        return self.X[:2].flatten()


def track(results):
    res = results[0]
    bboxes = []
    for box in res.boxes:
        bboxes.append(box.xyxy[0])
    bboxes = np.array(bboxes)