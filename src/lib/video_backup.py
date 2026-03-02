import json, cv2, glob, os, numpy as np 
from ultralytics import YOLO
import time

def play_video():
    model = YOLO("/home/deveshdatwani/Downloads/yolov8n.pt")
    img_dir = "/home/deveshdatwani/mot/data/a9_dataset_r02_s01/images/s110_camera_basler_south2_8mm"
    json_dir = "/home/deveshdatwani/mot/data/a9_dataset_r02_s01/labels_point_clouds/s110_lidar_ouster_south"
    img_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    json_files = sorted(glob.glob(os.path.join(json_dir, "*.json")))
    GLOBAL = 0
    p_time = time.time()
    new_time = 0

    def draw_fps(frame, prev_time):
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        cv2.putText(frame, f"frame number {GLOBAL}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"{fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        return curr_time

    def get_camera_K(cam_name, metadata):
        cam = metadata["openlabel"]["streams"][cam_name]["stream_properties"]["intrinsics_pinhole"]["camera_matrix_3x4"]
        return np.array(cam)[:3,:3]

    def get_lidar_to_cam(cam_name, metadata):
        pose = metadata["openlabel"]["coordinate_systems"][cam_name]["pose_wrt_parent"]["matrix4x4"]
        return np.array(pose).reshape((4,4))

    def project_points(corners, T):
        corners_cam = (T @ corners.T).T[:, :3]
        uv = (K @ corners_cam.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        return uv.astype(int)

    prev_time = 0
    
    for img_path in img_files:
        ts = float(os.path.basename(img_path).split("_")[0])
        closest_json = min(json_files, key=lambda f: abs(float(os.path.basename(f).split("_")[0]) - ts))
        
        with open(closest_json) as f:
            metadata = json.load(f)
        
        cam_name = "s110_camera_basler_south2_8mm"
        K = get_camera_K(cam_name, metadata)
        T_lidar_cam = get_lidar_to_cam(cam_name, metadata)
        
        img = cv2.imread(img_path)
        
        results = model(img)
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{model.names[cls]} {conf:.2f}"
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        prev_time = draw_fps(img, prev_time)
        new_time = time.time()
        GLOBAL += (new_time - p_time) 
        p_time = new_time
        frame_id = list(metadata["openlabel"]["frames"].keys())[0]
        objects = metadata["openlabel"]["frames"][frame_id]["objects"]
        
        for obj_id, obj_data in objects.items():
            cuboid = obj_data["object_data"]["cuboid"]["val"]
            x, y, z = cuboid[0:3]
            yaw, pitch, roll = cuboid[3:6]
            l, w, h = cuboid[7:10]

            dx, dy, dz = l/2, w/2, h/2
            corners = np.array([
                [ dx,  dy,  dz, 1],[ dx, -dy,  dz, 1],[-dx, -dy,  dz, 1],[-dx,  dy,  dz, 1],
                [ dx,  dy, -dz, 1],[ dx, -dy, -dz, 1],[-dx, -dy, -dz, 1],[-dx,  dy, -dz, 1]
            ]) + np.array([x, y, z, 0])
            
            uv = project_points(corners, T_lidar_cam)
            x1, y1 = uv[:,0].min(), uv[:,1].min()
            x2, y2 = uv[:,0].max(), uv[:,1].max()
            cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
        
        cv2.imshow("img", img)
        if cv2.waitKey(1) & 0xFF == ord('q'): break