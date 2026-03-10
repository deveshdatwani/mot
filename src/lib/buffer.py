import time

class Buffer:
    def __init__(self, max_size=30000):
        self.max_size = max_size
        self.data = []

    def update(self, frame):
        time_value = {"timestamp": time.time(), 
                      "frame": frame}
        self.data.append(time_value)