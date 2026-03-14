import cv2, time, json
import os, json, numpy as np

def draw_yolov8_detections(img, results):
    res = results[0]
    for box in res.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        cls = int(box.cls[0].item())
        conf = box.conf[0].item()
        draw_corner_box(img, (x1, y1), (x2, y2), (0, 255, 0), 1, 20)
        label = f"{res.names[cls]} {conf:.2f}"
        if box.id is not None:
            label = f"ID: {int(box.id.item())} {label}"
        cv2.putText(img, label, (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
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

def draw_corner_box(img, pt1, pt2, color, thickness, length=20):
    length = int(0.3 * (pt2[1] - pt1[1]))
    x1, y1 = pt1
    x2, y2 = pt2
    cv2.line(img, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(img, (x1, y1), (x1, y1 + length), color, thickness)
    cv2.line(img, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + length), color, thickness)
    cv2.line(img, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - length), color, thickness)
    cv2.line(img, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - length), color, thickness)

def draw_fps(frame, ts, prev_time):
    fps = 1 / (ts - prev_time) if prev_time != 0 else 0
    color = (0, 255, 255) if fps > 8 else (0, 0, 255)
    cv2.putText(frame, f"FPS: {fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    prev_time = ts
    return prev_time, frame

def draw_custom_tracks(img, object_log):
    from lib.drawing import draw_corner_box
    for pid, t in object_log.items():
        x1, y1, x2, y2 = map(int, t.bbox)
        color = (0, 255, 0)
        draw_corner_box(img, (x1, y1), (x2, y2), color, 2, 20)
        cv2.putText(img, f"ID: {pid}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        curr_pos = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        if len(t.history) == 0 or np.linalg.norm(np.array(t.history[-1]) - curr_pos) > 2:
            t.history.append(curr_pos)
        if len(t.history) > 40: t.history.pop(0)
        if len(t.history) > 1:
            cv2.polylines(img, [np.array(t.history, np.int32).reshape((-1, 1, 2))], False, color, 1)
        for f in t.features:
            cv2.circle(img, (int(f.pos[0]), int(f.pos[1])), 2, (0, 255, 255), -1)

def draw_yolo_seg(img, results, object_log):
    if results[0].masks is not None:
        mask_img = results[0].plot(labels=False, boxes=False)
        cv2.addWeighted(mask_img, 0.4, img, 0.6, 0, img)
    for pid, t in object_log.items():
        x1, y1, x2, y2 = map(int, t.bbox)
        draw_corner_box(img, (x1, y1), (x2, y2), t.color, 2)
        cv2.putText(img, f"ID: {pid}", (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, t.color, 2)
        if len(t.history) > 1:
            cv2.polylines(img, [np.array(t.history, np.int32).reshape((-1,1,2))], False, t.color, 2)
    return img

TRACK_COLORS = [
    (0, 255, 0), (0, 0, 255), (0, 0, 255), 
    (255, 255, 0), (0, 255, 255), (255, 0, 255)
]

def draw_fyveby_gt(frame, frame_idx, annotations):  
    for ann in annotations:
        tid = ann["track_id"]
        x1, y1, x2, y2 = [int(v) for v in ann["bbox"]]
        color = TRACK_COLORS[tid % len(TRACK_COLORS)]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"ID:{tid} {ann.get('class', 'aircraft')}"
        cv2.putText(
            frame, label, (x1, y1 - 8), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
    return frame