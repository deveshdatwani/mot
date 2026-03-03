import cv2
import os, json, numpy as np


def draw_yolov8_detections(img, results):
    res = results[0]
    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0].item())
        conf = box.conf[0].item()
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"{res.names[cls]} {conf:.2f}"
        if box.id is not None:
            label = f"ID: {int(box.id.item())} {label}"
        cv2.putText(img, label, (x1, y1 + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img

def project_points(corners, T, K):
    corners_cam = (T @ corners.T).T[:, :3]
    uv = (K @ corners_cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv.astype(int)

def project_lidar_on_camera(img, T_lidar_cam, K, img_path, json_files):
    ts=float(os.path.basename(img_path).split("_")[0])
    closest_json=min(json_files,key=lambda f:abs(float(os.path.basename(f).split("_")[0])-ts))
    with open(closest_json) as f:
        data=json.load(f)
    frames=data["openlabel"]["frames"]
    frame_id=list(frames.keys())[0]
    objects=frames[frame_id]["objects"]
    h_img,w_img=img.shape[:2]
    for obj_id,obj_data in objects.items():
        cuboid=obj_data["object_data"]["cuboid"]["val"]
        x,y,z=cuboid[0:3]
        l,w,h=cuboid[7:10]
        dx,dy,dz=l/2,w/2,h/2
        corners=np.array([[dx,dy,dz,1],[dx,-dy,dz,1],[-dx,-dy,dz,1],[-dx,dy,dz,1],[dx,dy,-dz,1],[dx,-dy,-dz,1],[-dx,-dy,-dz,1],[-dx,dy,-dz,1]])+np.array([x,y,z,0])
        uv=project_points(corners,T_lidar_cam,K)
        uv=uv[np.isfinite(uv).all(axis=1)]
        if len(uv)==0:continue
        x1,y1=np.min(uv[:,0]),np.min(uv[:,1])
        x2,y2=np.max(uv[:,0]),np.max(uv[:,1])
        x1=int(np.clip(x1,0,w_img-1));x2=int(np.clip(x2,0,w_img-1))
        y1=int(np.clip(y1,0,h_img-1));y2=int(np.clip(y2,0,h_img-1))
        cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
    return None