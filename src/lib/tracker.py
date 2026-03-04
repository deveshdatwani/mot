import numpy as np
import cv2
class KalmanFilter:
    def __init__(self, x, y):
        self.dt = 1/30
        self.X = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.A = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * 1.0
        self.R = np.eye(2, dtype=np.float32) * 0.1
        self.P = np.eye(4, dtype=np.float32)
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
        self.P = (np.eye(4) - (K @ self.H)) @ self.P
        self.miss_count = 0
        return self.X[:2].flatten()

def track(results, img, object_log, next_id, max_miss=30):
    res = results[0]
    bboxes = res.boxes.xyxy.cpu().numpy() if len(res.boxes) > 0 else []
    centers = np.stack((bboxes[:, 0] + (bboxes[:, 2]-bboxes[:, 0])/2, bboxes[:, 1] + (bboxes[:, 3]-bboxes[:, 1])/2), axis=1) if len(bboxes) > 0 else []
    new_object_log = {}
    matched_detections = {} 
    for obj_id, kf in object_log.items():
        pred = kf.predict()
        if len(centers) > 0:
            dists = np.linalg.norm(centers - pred, axis=1)
            idx = np.argmin(dists)
            if dists[idx] < 50:
                if idx not in matched_detections or dists[idx] < matched_detections[idx][1]:
                    matched_detections[idx] = (obj_id, dists[idx])
    for idx, (obj_id, dist) in matched_detections.items():
        kf = object_log[obj_id]
        kf.update(centers[idx])
        kf.history.append((int(kf.X[0,0]), int(kf.X[1,0])))
        if len(kf.history) > 20: kf.history.pop(0)
        new_object_log[obj_id] = kf
    for obj_id, kf in object_log.items():
        if obj_id not in new_object_log:
            kf.miss_count += 1
            if kf.miss_count <= max_miss:
                kf.history.append((int(kf.X[0,0]), int(kf.X[1,0])))
                if len(kf.history) > 20: kf.history.pop(0)
                new_object_log[obj_id] = kf
    for i, pos in enumerate(centers):
        if i not in matched_detections:
            new_object_log[next_id] = KalmanFilter(pos[0], pos[1])
            next_id += 1
    for obj_id, kf in new_object_log.items():
        p = kf.X[:2].flatten()
        c = (0, 255, 0) if kf.miss_count == 0 else (0, 0, 255)
        cv2.putText(img, f"ID: {obj_id}", (int(p[0]), int(p[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c, 2)
        for j in range(1, len(kf.history)):
            cv2.line(img, kf.history[j-1], kf.history[j], c, 2)
    return new_object_log, next_id