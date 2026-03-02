import numpy as np
import cv2

class SimpleTracker:
    def __init__(self, dist_thresh=200, max_missed=5):
        self.dist_thresh = dist_thresh
        self.max_missed = max_missed
        self.next_id = 0
        self.tracks = {}  # id -> {"centroid":(x,y), "pts":[...], "missed":0}

    def update(self, detections):  # detections = [[x1,y1,x2,y2], ...]
        centroids = []
        for x1,y1,x2,y2 in detections:
            centroids.append(np.array([(x1+x2)/2, (y1+y2)/2]))

        used = set()
        for tid in list(self.tracks.keys()):
            prev_c = self.tracks[tid]["centroid"]
            if len(centroids) == 0:
                self.tracks[tid]["missed"] += 1
                continue

            dists = np.linalg.norm(np.array(centroids) - prev_c, axis=1)
            idx = np.argmin(dists)

            if dists[idx] < self.dist_thresh and idx not in used:
                self.tracks[tid]["centroid"] = centroids[idx]
                self.tracks[tid]["pts"].append(tuple(centroids[idx].astype(int)))
                self.tracks[tid]["missed"] = 0
                used.add(idx)
            else:
                self.tracks[tid]["missed"] += 1

        # create new tracks
        for i,c in enumerate(centroids):
            if i not in used:
                self.tracks[self.next_id] = {
                    "centroid": c,
                    "pts": [tuple(c.astype(int))],
                    "missed": 0
                }
                self.next_id += 1

        # delete dead tracks
        for tid in list(self.tracks.keys()):
            if self.tracks[tid]["missed"] > self.max_missed:
                del self.tracks[tid]

        return self.tracks