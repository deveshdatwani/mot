#!/home/deveshdatwani/mot/venv/bin/python3
import json, cv2, glob, os, numpy as np 
from ultralytics import YOLO
from lib.a9 import get_camera_K, get_lidar_to_cam
from lib.drawing import draw_yolov8_detections, draw_fps, draw_custom_tracks, draw_fyveby_gt
from lib.tracker import track_airplanes as track
import time

model = YOLO("/home/deveshdatwani/Downloads/yolov8s.pt")
img_dir = "/home/deveshdatwani/mot/data/a9_dataset_r02_s01/images/s110_camera_basler_south2_8mm"
json_dir = "/home/deveshdatwani/mot/data/a9_dataset_r02_s01/labels_point_clouds/s110_lidar_ouster_south"
img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
with open("/home/deveshdatwani/Downloads/cv_coding_challenge/cv_coding_challenge/annotations/simple_plane_add.json") as f:
    fyve_by_json = json.load(f)["frames"]

if __name__ == "__main__":
    cap = cv2.VideoCapture("/home/deveshdatwani/Downloads/cv_coding_challenge/cv_coding_challenge/data/simple_plane_add.mp4")
    object_log, next_id = {}, 0
    prev_time = 0
    prev_gray = None
    
    meta = json.load(open(json_files[0]))
    T_lidar_cam = get_lidar_to_cam("s110_camera_basler_south2_8mm", meta)
    K = get_camera_K("s110_camera_basler_south2_8mm", meta)
    annotations_idx = list(fyve_by_json.keys())
    i = 0 

    while True:
        ts = time.time()
        ret, img = cap.read()
        if not ret: break
        results = model(img, verbose=False)
        object_log, next_id, prev_gray = track(results, img, prev_gray, object_log, next_id )
        draw_yolov8_detections(img, results)
        draw_custom_tracks(img, object_log)
        draw_fyveby_gt(img, i, fyve_by_json.get(annotations_idx[i]))
        # draw_yolo_seg(img, results, object_log)
        prev_time, img = draw_fps(img, ts, prev_time)
        display_img = cv2.resize(img, (1280, 720))
        cv2.imshow("img", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break
        i += 1
    cap.release()
    cv2.destroyAllWindows()