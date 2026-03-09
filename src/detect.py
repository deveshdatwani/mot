#!/home/deveshdatwani/mot/venv/bin/python3
import json, cv2, glob, os, numpy as np 
from ultralytics import YOLO
import time
from lib.a9 import get_camera_K, get_lidar_to_cam, project_points
from lib.fps import draw_fps    
from lib.drawing import draw_yolov8_detections, project_lidar_on_camera
from lib.tracker import track
from lib.rpi_cam import capture_rpi_camera
from queue import Queue
from threading import Thread

model = YOLO("/home/deveshdatwani/Downloads/yolov8n.pt")
img_dir = "/home/deveshdatwani/mot/data/a9_dataset_r02_s01/images/s110_camera_basler_south2_8mm"
json_dir = "/home/deveshdatwani/mot/data/a9_dataset_r02_s01/labels_point_clouds/s110_lidar_ouster_south"
frame_queue = Queue(maxsize=1) 
img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
GLOBAL = 0
thread = Thread(target=capture_rpi_camera, args=(frame_queue,))
thread.start()
prev_time = 0
object_log, next_id = {}, 0
tracks = {}

if __name__ == "__main__":
    meta = json.load(open(json_files[0]))
    T_lidar_cam = get_lidar_to_cam("s110_camera_basler_south2_8mm", meta)
    K = get_camera_K("s110_camera_basler_south2_8mm", meta)
    while True:
        ts, img = frame_queue.get()
        results = model(img, verbose=False, conf=0.4)
        draw_yolov8_detections(img, results)
        prev_time = draw_fps(img, prev_time)
        tracks, next_id = track(results, img, ts, tracks, next_id, 50)
        img = cv2.resize(img,   (1280, 720))
        cv2.imshow("img", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()
