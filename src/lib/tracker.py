import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment

class KalmanFilter:
    def __init__(self, x, y, timestamp):
        self.last_time = timestamp
        self.X = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.R = np.eye(2, dtype=np.float32) * 0.1 
        self.P = np.eye(4, dtype=np.float32) * 100.0
        self.miss_count = 0
        self.history = []

    def predict(self, current_time):
        dt = current_time - self.last_time
        self.last_time = current_time
        A = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ], dtype=np.float32)
        q_weight = 15.0
        Q = np.eye(4, dtype=np.float32) * q_weight * dt
        self.X = A @ self.X
        self.P = (A @ self.P @ A.T) + Q
        return self.X[:2].flatten()

    def update(self, z):
        z = np.array(z, dtype=np.float32).reshape((2, 1))
        S = (self.H @ self.P @ self.H.T) + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.X = self.X + (K @ (z - (self.H @ self.X)))
        self.P = (np.eye(4, dtype=np.float32) - (K @ self.H)) @ self.P
        self.miss_count = 0
        return self.X[:2].flatten()

def track(results, img, current_timestamp, object_log, next_id, max_miss=60):
    res = results[0]
    bboxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) > 0 else np.empty((0, 4))
    centers = np.column_stack([
        bboxes[:, 0] + (bboxes[:, 2] - bboxes[:, 0]) / 2, 
        bboxes[:, 1] + (bboxes[:, 3] - bboxes[:, 1]) / 2
    ])
    obj_ids = list(object_log.keys())
    new_object_log = {}
    matched_det_indices = set()
    if len(obj_ids) > 0:
        preds = np.array([object_log[oid].predict(current_timestamp) for oid in obj_ids])
        if len(centers) > 0:
            dist_matrix = np.linalg.norm(preds[:, np.newaxis, :] - centers[np.newaxis, :, :], axis=2)
            rows, cols = linear_sum_assignment(dist_matrix)            
            for r, c in zip(rows, cols):
                if dist_matrix[r, c] < 50:
                    oid = obj_ids[r]
                    kf = object_log[oid]
                    kf.update(centers[c])
                    kf.history.append((int(kf.X[0,0]), int(kf.X[1,0])))
                    if len(kf.history) > 30: kf.history.pop(0)
                    new_object_log[oid] = kf
                    matched_det_indices.add(c)
    for oid in obj_ids:
        if oid not in new_object_log:
            kf = object_log[oid]
            kf.miss_count += 1
            if kf.miss_count <= max_miss:
                kf.history.append((int(kf.X[0,0]), int(kf.X[1,0])))
                if len(kf.history) > 30: kf.history.pop(0)
                new_object_log[oid] = kf
    for i, pos in enumerate(centers):
        if i not in matched_det_indices:
            new_object_log[next_id] = KalmanFilter(pos[0], pos[1], current_timestamp)
            next_id += 1
    for oid, kf in new_object_log.items():
        p = kf.X[:2].flatten()
        color = (0, 255, 0) if kf.miss_count == 0 else (0, 0, 255)
        cv2.putText(img, f"ID: {oid}", (int(p[0]), int(p[1]) - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        if len(kf.history) > 1:
            cv2.polylines(img, [np.array(kf.history).reshape((-1, 1, 2))], False, color, 2)
    current_ids = list(new_object_log.keys())
    states = np.array([kf.X.flatten() for kf in new_object_log.values()])
    if len(states) > 1:
        dist_matrix = np.linalg.norm(states[:, np.newaxis, :2] - states[np.newaxis, :, :2], axis=2)
        np.fill_diagonal(dist_matrix, np.inf)
        rows, cols = np.where(dist_matrix < 10)
        merged_indices = set()
        for r, c in zip(rows, cols):
            if r not in merged_indices and c not in merged_indices:
                primary_id = current_ids[r]
                duplicate_id = current_ids[c]
                if new_object_log[duplicate_id].miss_count < new_object_log[primary_id].miss_count:
                    new_object_log[primary_id].miss_count = new_object_log[duplicate_id].miss_count
                new_object_log.pop(duplicate_id)
                merged_indices.add(c)
    return new_object_log, next_id