import cv2
import time

def draw_fps(frame, prev_time):
    curr_time = time.time()
    fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
    cv2.putText(frame, f"{fps:.1f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    return curr_time