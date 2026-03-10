from flask import Flask, Response, render_template
import cv2
import time

app = Flask(__name__)
rtsp_url = "rtsp://192.168.1.79:8554/cam"
camera = cv2.VideoCapture(rtsp_url)

def generate_frames():
    while True:
        success, frame = camera.read()
        frame = cv2.resize(frame, (640, 480))
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return b"welcome to mot system"

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)