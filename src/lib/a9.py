import numpy as np
import cv2
import os, json


def get_camera_K(cam_name, metadata):
    cam = metadata["openlabel"]["streams"][cam_name]["stream_properties"]["intrinsics_pinhole"]["camera_matrix_3x4"]
    return np.array(cam)[:3,:3]

def get_lidar_to_cam(cam_name, metadata):
    pose = metadata["openlabel"]["coordinate_systems"][cam_name]["pose_wrt_parent"]["matrix4x4"]
    return np.array(pose).reshape((4,4))

def project_points(corners, T, K):
    corners_cam = (T @ corners.T).T[:, :3]
    uv = (K @ corners_cam.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    return uv