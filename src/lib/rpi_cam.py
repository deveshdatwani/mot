import cv2
import time

rtsp_url = "rtsp://192.168.1.79:8554/cam"
cap = cv2.VideoCapture(rtsp_url)

def capture_rpi_camera(frame_queue):
    while True:
        ret, frame = cap.read()
        current_time = time.time()
        if not ret:
            continue
        if frame_queue.full():
            frame_queue.get() 
        frame_queue.put((current_time, frame))
    return frame_queue