import numpy as np
import cv2
from lib.nms import nms

class Feature:
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=np.float32)
        self.age = 0

    def update(self, new_x, new_y):
        self.pos = np.array([new_x, new_y], dtype=np.float32)
        self.age += 1

class Feature:
    def __init__(self, x, y):
        self.pos = np.array([x, y], dtype=np.float32)
        self.age = 0

    def update(self, new_x, new_y):
        self.pos = np.array([new_x, new_y], dtype=np.float32)
        self.age += 1

class PlaneTracker:
    def __init__(self, plane_id, bbox, gray_img):
        self.id = plane_id
        self.bbox = bbox
        self.features = []
        self.miss_count = 0
        self.history = [] 
        cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
        self.state = np.array([cx, cy, 0, 0], dtype=np.float32)
        self.uncertainty = np.eye(4, dtype=np.float32) * 100.0        
        self.seed_features(bbox, gray_img)

    def predict_physics(self, dt=1.0):
        A = np.array([[1, 0, dt, 0],
                      [0, 1, 0, dt],
                      [0, 0, 1, 0],
                      [0, 0, 0, 1]], dtype=np.float32)
        Q = np.eye(4, dtype=np.float32) * 0.1        
        self.state = A @ self.state
        self.uncertainty = A @ self.uncertainty @ A.T + Q

    def update_physics(self, measured_cx, measured_cy):
        H = np.array([[1, 0, 0, 0],
                      [0, 1, 0, 0]], dtype=np.float32)
        R = np.eye(2, dtype=np.float32) * 2.0
        z = np.array([measured_cx, measured_cy], dtype=np.float32)
        S = H @ self.uncertainty @ H.T + R
        K = self.uncertainty @ H.T @ np.linalg.inv(S)
        y = z - (H @ self.state)
        self.state = self.state + (K @ y)
        self.uncertainty = (np.eye(4) - (K @ H)) @ self.uncertainty

    def seed_features(self, bbox, gray_img):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = gray_img.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1: return
        
        mask = np.zeros_like(gray_img)
        mask[y1:y2, x1:x2] = 255
        pts = cv2.goodFeaturesToTrack(gray_img, maxCorners=20, qualityLevel=0.05, minDistance=5, mask=mask)
        if pts is not None:
            for p in pts:
                self.features.append(Feature(p[0][0], p[0][1]))

    def get_feature_consensus(self):
        if not self.features: return None
        pts = np.array([f.pos for f in self.features])
        weights = np.array([f.age + 1 for f in self.features])
        median = np.median(pts, axis=0)
        dist = np.linalg.norm(pts - median, axis=1)
        mask = dist < (np.std(dist) * 2 + 1)
        if not np.any(mask): return median
        return np.average(pts[mask], axis=0, weights=weights[mask])

    def get_smoothed_pos(self):
        """Returns the mean of the last 5 history points for visual smoothing."""
        if len(self.history) < 5:
            return self.state[0], self.state[1]
        last_pts = np.array(self.history[-5:])
        return np.mean(last_pts, axis=0)

def track_airplanes(results, frame, prev_gray, object_log, next_id):
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    for pid, tracker in object_log.items():
        tracker.predict_physics() 
        if tracker.features and prev_gray is not None:
            p0 = np.array([f.pos for f in tracker.features], dtype=np.float32)
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, winSize=(21,21), maxLevel=3)
            tracker.features = [tracker.features[i] for i, s in enumerate(st) if s == 1]
            for i, f in enumerate(tracker.features):
                f.update(p1[st.flatten()==1][i][0], p1[st.flatten()==1][i][1])
            meas = tracker.get_feature_consensus()
            if meas is not None:
                tracker.update_physics(meas[0], meas[1])
    res = results[0]
    air_idx = [i for i, c in enumerate(res.boxes.cls.cpu().numpy()) if res.names[int(c)].lower() == 'airplane']
    curr_bboxes = res.boxes.xyxy.cpu().numpy()[air_idx]    
    matched_ids = set()
    for bbox in curr_bboxes:
        best_id, max_age = -1, -1
        for pid, tracker in object_log.items():
            if pid in matched_ids: continue
            in_box = [f for f in tracker.features if bbox[0]<=f.pos[0]<=bbox[2] and bbox[1]<=f.pos[1]<=bbox[3]]
            if in_box:
                age_sum = sum(f.age for f in in_box)
                if age_sum > max_age:
                    max_age, best_id = age_sum, pid
        if best_id != -1:
            tracker = object_log[best_id]
            cx, cy = (bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2
            tracker.update_physics(cx, cy)
            tracker.bbox, tracker.miss_count = bbox, 0
            if len(tracker.features) < 15: tracker.seed_features(bbox, curr_gray)
            matched_ids.add(best_id)
        else:
            is_duplicate = False
            for pid in matched_ids:
                b = object_log[pid].bbox
                inter_x1, inter_y1 = max(bbox[0], b[0]), max(bbox[1], b[1])
                inter_x2, inter_y2 = min(bbox[2], b[2]), min(bbox[3], b[3])
                inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
                if inter_area > 0:
                    is_duplicate = True
                    break
            if not is_duplicate:
                object_log[next_id] = PlaneTracker(next_id, bbox, curr_gray)
                next_id += 1
    final_log = {}
    for pid, tracker in object_log.items():
        if pid not in matched_ids: tracker.miss_count += 1
        if tracker.miss_count < 40 and len(tracker.features) > 2:
            final_log[pid] = tracker
            tracker.history.append((tracker.state[0], tracker.state[1]))
            if len(tracker.history) > 60: tracker.history.pop(0) # Longer context
            sx, sy = tracker.get_smoothed_pos()
            clr = (0, 255, 0) if tracker.miss_count == 0 else (0, 165, 255)
            cv2.putText(frame, f"PLANE {pid} (v:{np.linalg.norm(tracker.state[2:]):.1f}px/f)", 
                        (int(sx), int(sy)-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, clr, 2)
            if len(tracker.history) > 2:
                pts = np.array(tracker.history, np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [pts], False, clr, 2, cv2.LINE_AA)                
    return final_log, next_id, curr_gray