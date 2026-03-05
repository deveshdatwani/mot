import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment


class KalmanFilter:
    def __init__(self, x, y):
        self.dt = 1/30
        self.X = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.A = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * 15.0
        self.R = np.eye(2, dtype=np.float32) * 0.01
        self.P = np.eye(4, dtype=np.float32) * 100.0
        self.miss_count = 0
        self.history = []
    def predict(self):
        self.X = self.A @ self.X
        self.P = (self.A @ self.P @ self.A.T) + self.Q
        return self.X[:2].flatten()
    def update(self, z):
        z = np.array(z, dtype=np.float32).reshape((2, 1))
        S = (self.H @ self.P @ self.H.T) + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.X = self.X + (K @ (z - (self.H @ self.X)))
        self.P = (np.eye(4, dtype=np.float32) - (K @ self.H)) @ self.P
        self.miss_count = 0
        return self.X[:2].flatten()

def track(results, img, object_log, next_id, max_miss=60):
    res = results[0]
    bboxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) > 0 else np.empty((0, 4))
    centers = np.column_stack([bboxes[:, 0] + (bboxes[:, 2] - bboxes[:, 0]) / 2, bboxes[:, 1] + (bboxes[:, 3] - bboxes[:, 1]) / 2])
    obj_ids = list(object_log.keys())
    new_object_log = {}
    matched_det_indices = set()
    if len(obj_ids) > 0 and len(centers) > 0:
        preds = np.array([object_log[oid].predict() for oid in obj_ids])
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
            new_object_log[next_id] = KalmanFilter(pos[0], pos[1])
            next_id += 1
    for oid, kf in new_object_log.items():
        p, c = kf.X[:2].flatten(), (0, 255, 0) if kf.miss_count == 0 else (0, 0, 255)
        cv2.putText(img, f"ID: {oid}", (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        if len(kf.history) > 1:
            cv2.polylines(img, [np.array(kf.history).reshape((-1, 1, 2))], False, c, 2)
    return new_object_log, next_id