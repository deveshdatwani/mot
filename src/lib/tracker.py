import numpy as np
import cv2


class KalmanFilter:
    def __init__(self, x, y):
        self.dt = 1/30
        self.X = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.A = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * 0.01
        self.R = np.eye(2, dtype=np.float32) * 0.1
        self.P = np.eye(4, dtype=np.float32)
        self.miss_count = 0
    
    def predict(self):
        self.X = self.A @ self.X
        self.P = (self.A @ self.P @ self.A.T) + self.Q
        return self.X[:2].flatten()
    
    def update(self, z):
        z = np.array(z, dtype=np.float32).reshape((2, 1))
        S = (self.H @ self.P @ self.H.T) + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.X = self.X + (K @ (z - (self.H @ self.X)))
        self.P = (np.eye(4) - (K @ self.H)) @ self.P
        self.miss_count = 0
        return self.X[:2].flatten()

def track(results, img, object_log, next_id, max_miss=20):
    res = results[0]
    bboxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) > 0 else []
    centers = np.stack((bboxes[:, 0] + (bboxes[:, 2]-bboxes[:, 0])/2, bboxes[:, 1] + (bboxes[:, 3]-bboxes[:, 1])/2), axis=1) if len(bboxes) > 0 else []
    new_object_log = {}
    matched_indices = set() 
    for obj_id, kf in object_log.items():
        pred = kf.predict()
        best_match_idx = -1
        if len(centers) > 0:
            dists = np.linalg.norm(centers - pred, axis=1)
            best_match_idx = np.argmin(dists)
            if dists[best_match_idx] < 100:
                kf.update(centers[best_match_idx])
                matched_indices.add(best_match_idx)
            else:
                kf.miss_count += 1
        else:
            kf.miss_count += 1
        if kf.miss_count <= max_miss:
            new_object_log[obj_id] = kf
    for i, obj_pos in enumerate(centers):
        if i not in matched_indices:
            new_object_log[next_id] = KalmanFilter(obj_pos[0], obj_pos[1])
            next_id += 1 
    for obj_id, kf in new_object_log.items():
        pos = kf.X[:2].flatten()
        color = (0, 255, 0) if kf.miss_count == 0 else (0, 0, 255)
        cv2.putText(img, f"ID: {obj_id}", (int(pos[0]), int(pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return new_object_log, next_id