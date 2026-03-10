#!/home/deveshdatwani/mot/venv/bin/python3
import json, cv2, glob, os, numpy as np 
from ultralytics import YOLO
from lib.a9 import get_camera_K, get_lidar_to_cam, project_points 
from lib.drawing import draw_yolov8_detections, draw_fps
from lib.tracker import track
from lib.rpi_cam import capture_rpi_camera
from queue import Queue
from threading import Thread
from lib.buffer import Buffer

GLOBAL = 0
frame_queue = Queue(maxsize=1) 
model = YOLO("/home/deveshdatwani/Downloads/yolov8n.pt")
img_dir = "/home/deveshdatwani/mot/data/a9_dataset_r02_s01/images/s110_camera_basler_south2_8mm"
json_dir = "/home/deveshdatwani/mot/data/a9_dataset_r02_s01/labels_point_clouds/s110_lidar_ouster_south"
img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
thread = Thread(target=capture_rpi_camera, args=(frame_queue,))

if __name__ == "__main__":
    object_log, next_id = {}, 0
    tracks = {}
    buffer = Buffer()
    thread.start()
    prev_time = 0
    meta = json.load(open(json_files[0]))
    T_lidar_cam = get_lidar_to_cam("s110_camera_basler_south2_8mm", meta)
    K = get_camera_K("s110_camera_basler_south2_8mm", meta)
    while True:
        ts, img = frame_queue.get()
        results = model(img, verbose=False)
        draw_yolov8_detections(img, results)
        tracks, next_id = track(results, img, ts, tracks, next_id, 50)
        prev_time = draw_fps(img, ts, prev_time)
        img = cv2.resize(img,   (1280, 720))
        cv2.imshow("img", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
    cv2.destroyAllWindows()