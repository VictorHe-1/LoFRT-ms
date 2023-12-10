import time


class Timer:
    def __init__(self, name=''):
        self.name = name
        self.start = 0.0
        self.end = 0.0

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end = time.time()
        print(f'{self.name} cost time {self.end - self.start:.3f}')