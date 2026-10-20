import time

class Timer:
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.seconds = time.time() - self.start
        print(f"Time taken: {self.seconds} seconds")