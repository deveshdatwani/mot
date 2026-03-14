import numpy as np
import cv2
from lib.nms import nms

class Feature:
    def __init__(self, x, y, score):
        self.pos = np.array([x, y], dtype=np.float32)
        self.score = score  # The corner quality score
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
        self.seed_features(bbox, gray_img)

    def seed_features(self, bbox, gray_img):
        x1, y1, x2, y2 = map(int, bbox)
        h, w = gray_img.shape
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1: return
        
        mask = np.zeros_like(gray_img)
        mask[y1:y2, x1:x2] = 255
        
        # goodFeaturesToTrack returns points sorted by quality score descending
        # We limit to 20 high-score points as requested
        pts = cv2.goodFeaturesToTrack(gray_img, maxCorners=20, qualityLevel=0.05, minDistance=5, mask=mask)
        
        if pts is not None:
            # We don't want to double-count features if we are re-seeding
            existing_pts = np.array([f.pos for f in self.features]) if self.features else np.empty((0,2))
            
            for p in pts:
                new_p = p[0]
                if len(existing_pts) > 0:
                    dist = np.linalg.norm(existing_pts - new_p, axis=1)
                    if np.any(dist < 5): continue # Skip if too close to an existing tracked feature
                
                self.features.append(Feature(new_p[0], new_p[1], score=1.0))
            
            # Final prune to ensure we never exceed 20 total
            if len(self.features) > 20:
                self.features.sort(key=lambda x: x.age, reverse=True) # Keep the ones we've tracked longest
                self.features = self.features[:20]

    def get_weighted_center(self):
        if not self.features: return None
        pts = np.array([f.pos for f in self.features])
        # Weight heavily by age to trust established points over new seeds
        weights = np.array([f.age + 1 for f in self.features])
        
        median = np.median(pts, axis=0)
        dist = np.linalg.norm(pts - median, axis=1)
        mask = dist < (np.std(dist) * 2 + 1)
        
        if not np.any(mask): return median
        return np.average(pts[mask], axis=0, weights=weights[mask])

def track_airplanes(results, frame, prev_gray, object_log, next_id):
    curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # 1. Optical Flow on High-Score Features
    if prev_gray is not None and object_log:
        for tracker in object_log.values():
            if not tracker.features: continue
            
            p0 = np.array([f.pos for f in tracker.features], dtype=np.float32)
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, p0, None, winSize=(21,21), maxLevel=3)
            
            # Filter by status and error (discard high-error flow)
            valid_features = []
            for i, status in enumerate(st):
                if status == 1 and err[i] < 15: # Threshold error to discard drifting points
                    tracker.features[i].update(p1[i][0], p1[i][1])
                    valid_features.append(tracker.features[i])
            tracker.features = valid_features
            
            center = tracker.get_weighted_center()
            if center is not None and tracker.bbox is not None:
                w, h = tracker.bbox[2]-tracker.bbox[0], tracker.bbox[3]-tracker.bbox[1]
                tracker.bbox = [center[0]-w/2, center[1]-h/2, center[0]+w/2, center[1]+h/2]

    # 2. YOLO Detection & Association
    res = results[0]
    air_idx = [i for i, c in enumerate(res.boxes.cls.cpu().numpy()) if res.names[int(c)].lower() == 'airplane']
    curr_bboxes = res.boxes.xyxy.cpu().numpy()[air_idx]
    
    matched_ids = set()
    new_track_candidates = {}

    for bbox in curr_bboxes:
        best_id, max_age_sum = -1, -1
        for pid, tracker in object_log.items():
            if pid in matched_ids: continue
            
            # Count points inside the new detection box
            in_box = [f for f in tracker.features if bbox[0]<=f.pos[0]<=bbox[2] and bbox[1]<=f.pos[1]<=bbox[3]]
            if in_box:
                age_sum = sum(f.age for f in in_box)
                if age_sum > max_age_sum:
                    max_age_sum, best_id = age_sum, pid
        
        if best_id != -1:
            tracker = object_log[best_id]
            tracker.bbox, tracker.miss_count = bbox, 0
            # Replenish up to 20 if points were lost
            if len(tracker.features) < 15:
                tracker.seed_features(bbox, curr_gray)
            matched_ids.add(best_id)
        else:
            # Overlap check for new IDs
            is_overlapping = False
            for pid in matched_ids:
                b = object_log[pid].bbox
                iou = max(0, min(bbox[2], b[2]) - max(bbox[0], b[0])) * max(0, min(bbox[3], b[3]) - max(bbox[1], b[1]))
                if iou > 0:
                    is_overlapping = True; break
            
            if not is_overlapping:
                new_track_candidates[next_id] = PlaneTracker(next_id, bbox, curr_gray)
                next_id += 1

    object_log.update(new_track_candidates)
    final_log = {}
    
    # 3. Cleanup and Render
    for pid, tracker in object_log.items():
        if pid not in matched_ids and pid not in new_track_candidates:
            tracker.miss_count += 1
            
        if tracker.miss_count < 30 and len(tracker.features) > 2:
            final_log[pid] = tracker
            pos = tracker.get_weighted_center()
            if pos is not None:
                tracker.history.append((int(pos[0]), int(pos[1])))
                if len(tracker.history) > 30: tracker.history.pop(0)
                
                clr = (0, 255, 0) if tracker.miss_count == 0 else (0, 0, 255)
                cv2.putText(frame, f"PLANE {pid}", (int(pos[0]), int(pos[1])-15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, clr, 2)
                
                # Draw only the elite 20 features
                for f in tracker.features:
                    cv2.circle(frame, (int(f.pos[0]), int(f.pos[1])), 3, (255, 0, 255), -1)
                
                if len(tracker.history) > 1:
                    cv2.polylines(frame, [np.array(tracker.history).reshape((-1, 1, 2))], False, clr, 2)
    
    return final_log, next_id, curr_gray