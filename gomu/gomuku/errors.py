import sys
import threading
import queue

class BoardError(Exception):
    def __init__(self):
        super().__init__('Struggling the error when controlling the board.')

class RLEnvError(Exception):
    def __init__(self):
        super().__init__("Bad Response from Reinforcement Learning Environment.")
        
class BotError(Exception):
    def __init__(self):
        super().__init__("Bad Response from Bot during predicting the next position.")

class PosError(Exception):
    def __init__(self, col, row):
        super().__init__(f"Error has been occured while setting stone on ({col},{row})")

class ThreadingError(Exception):
    def __init__(self):
        super().__init__("Error occured in threading process.")

class ThreadingErrorHandler(threading.Thread):
    def __init__(self, bucket, **kwargs):
        super().__init__(**kwargs)
        self.bucket = bucket

    def run(self):
        try: 
            raise ThreadingError()
        except Exception:
            self.bucket.put(sys.exc_info())
class NanError(Exception):
    def __init__(self):
        super().__init__(f"NANANANANAN")