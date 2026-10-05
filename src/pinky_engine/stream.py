"""Single pending frame: bounded memory and no stale FIFO backlog."""
import threading


class LatestFrame:
    def __init__(self):
        self.condition = threading.Condition()
        self.pending = None
        self.closed = False
        self.dropped = 0

    def put(self, value):
        with self.condition:
            if self.closed:
                return False
            if self.pending is not None:
                self.dropped += 1
            self.pending = value
            self.condition.notify()
            return True

    def get(self):
        with self.condition:
            self.condition.wait_for(lambda: self.pending is not None or self.closed)
            value, self.pending = self.pending, None
            return value

    def close(self):
        with self.condition:
            self.closed = True
            self.condition.notify_all()
