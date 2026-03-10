import cv2
import time
from flask import Flask, render_template, send_from_directory

app = Flask(__name__)

@app.route('/video/')
def serve_video():
    return send_from_directory(
        '/home/deveshdatwani/mot/assets',
        'sample_tracking.mp4',
        conditional=True
    )

@app.route('/')
def index():
    return render_template('index.html')

                
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)