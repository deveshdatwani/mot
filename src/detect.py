import json, cv2, glob, os, numpy as np 
from ultralytics import YOLO
import time
from lib.tracker import SimpleTracker
from lib.a9 import get_camera_K, get_lidar_to_cam, project_points, project_lidar_on_camera
from lib.fps import draw_fps    
from lib.drawing import draw_detections

tracker = SimpleTracker()

model = YOLO("/home/deveshdatwani/Downloads/yolov8n.pt")

img_dir = "/home/deveshdatwani/mot/data/a9_dataset_r02_s01/images/s110_camera_basler_south2_8mm"
json_dir = "/home/deveshdatwani/mot/data/a9_dataset_r02_s01/labels_point_clouds/s110_lidar_ouster_south"

img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
GLOBAL = 0

prev_time = 0
curr = np.array([])

if __name__ == "__main__":
    for img_path in img_files:
        img = cv2.imread(img_path)
        results = model(img, verbose=False)
        draw_yolov8_detections(img, results)
        prev_time = draw_fps(img, prev_time)
        cv2.imshow("img", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cv2.destroyAllWindows()