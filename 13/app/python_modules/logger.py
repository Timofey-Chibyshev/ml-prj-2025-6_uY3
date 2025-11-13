import time
from datetime import datetime

class Logger(object):
    def __init__(self, frequency=100):
        self.start_time = time.time()
        self.frequency = frequency
        self.error_fn = None

    def __get_elapsed(self):
        return datetime.fromtimestamp(time.time() - self.start_time).strftime("%M:%S")

    def set_error_fn(self, error_fn):
        self.error_fn = error_fn
  
    def log_train_start(self, model):
        print("\nTraining started")
        print("================")

    def log_train_epoch(self, epoch, loss, custom="", is_iter=False):
        if epoch % self.frequency == 0:
            error_str = f"error: {self.error_fn():.4e}" if self.error_fn else ""
            print(f"{'nt_epoch' if is_iter else 'tf_epoch'} = {epoch:6d}  elapsed = {self.__get_elapsed()}  loss = {loss:.4e} {custom} {error_str}")

    def log_train_opt(self, name):
        print(f"—— Starting {name} optimization ——")

    def log_train_end(self, epoch, custom=""):
        print("==================")
        error_str = f"error = {self.error_fn():.4e}" if self.error_fn else ""
        print(f"Training finished (epoch {epoch}): duration = {self.__get_elapsed()} {error_str} {custom}")