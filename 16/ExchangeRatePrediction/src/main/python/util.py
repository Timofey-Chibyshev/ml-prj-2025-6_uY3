class LogProgressInfo:
    def __init__(self, process_name, progress, end):
        self.process_name = process_name
        self.progress = progress
        self.end = end

    def print(self):
        print(f"\r{self.process_name} progress: {self.progress}/{self.end}", end="", flush=True)

separate_line_length = 80
separate_sign = "="
def print_separate_message(s: str = "", sep: str = separate_sign):
    start_len = int((separate_line_length - len(s)) / 2) - 1
    end_len = separate_line_length - len(s) - start_len - 1
    print(f"{sep * start_len} {s} {sep * end_len}")

