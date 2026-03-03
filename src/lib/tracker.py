import numpy as np
import cv2

class KalmanFilter:
    def __init__(self, x, y):
        self.dt = 1/30
        self.X = np.array([[x], [y], [0], [0]], dtype=np.float32)
        self.A = np.array([[1, 0, self.dt, 0], [0, 1, 0, self.dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * 5.0
        self.R = np.eye(2, dtype=np.float32) * 5.0
        self.P = np.eye(4, dtype=np.float32)
    
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
        return self.X[:2].flatten()

def track(results, img, object_log, next_id):
    res = results[0]
    if len(res.boxes) == 0: return object_log, next_id
    bboxes = res.boxes.xyxy.cpu().numpy()
    centers = np.stack((bboxes[:, 0] + (bboxes[:, 2]-bboxes[:, 0])/2, bboxes[:, 1] + (bboxes[:, 3]-bboxes[:, 1])/2), axis=1)
    new_object_log = {}
    for obj_pos in centers:
        match_id, min_dist = None, 100
        for obj_id, kf in object_log.items():
            pred = kf.predict()
            dist = np.linalg.norm(pred - obj_pos)
            if dist < min_dist:
                min_dist, match_id = dist, obj_id
        if match_id is not None:
            kf = object_log[match_id]
            kf.update(obj_pos)
            new_object_log[match_id] = kf
            del object_log[match_id]
        else:
            new_object_log[next_id] = KalmanFilter(obj_pos[0], obj_pos[1])
            match_id = next_id
            next_id += 1
        cv2.putText(img, f"ID: {match_id}", (int(obj_pos[0]), int(obj_pos[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return new_object_log, next_id