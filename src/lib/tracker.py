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


def track(results, img, object_log, next_id):
    res = results[0]
    if len(res.boxes) == 0: return {}, next_id
    bboxes = res.boxes.xyxy.cpu().numpy()
    cx = bboxes[:, 0] + ((bboxes[:, 2] - bboxes[:, 0]) / 2)
    cy = bboxes[:, 1] + ((bboxes[:, 3] - bboxes[:, 1]) / 2)
    current_centers = np.stack((cx, cy), axis=1)
    new_object_log = {}
    temp_old_log = object_log.copy()
    for obj_pos in current_centers:
        match_id = None
        min_dist = 150
        for obj_id, prev_pos in temp_old_log.items():
            dist = np.linalg.norm(prev_pos - obj_pos)
            if dist < min_dist:
                min_dist = dist
                match_id = obj_id
        if match_id is not None:
            new_object_log[match_id] = obj_pos
            del temp_old_log[match_id]
        else:
            new_object_log[next_id] = obj_pos
            match_id = next_id
            next_id += 1
        cv2.putText(img, f"ID: {match_id}", (int(obj_pos[0]), int(obj_pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return new_object_log, next_id