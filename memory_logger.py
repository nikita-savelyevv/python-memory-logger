import atexit
import os
import queue
import subprocess
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt

import psutil


class MemoryLogger:
    def __init__(
            self,
            log_dir: Path,
            plot_title: str = "",
            interval: float = 1,
            memory_type: str = "rss",
            start_memory_from_zero: bool = True,
    ):
        assert memory_type in ["rss", "total-avail", "top"]
        self.log_dir = log_dir
        self.plot_title = plot_title
        self.interval = interval
        self.memory_type = memory_type
        self.start_memory_from_zero = start_memory_from_zero or memory_type == "total-avail"

        self.monitoring_thread_should_stop = False
        self.monitoring_in_progress = False

        self.memory_monitor_thread = None
        self.memory_data_queue = None
        self.stop_logging_atexit_fn = None

    def start_logging(self):
        if self.monitoring_in_progress:
            raise Exception("Monitoring already in progress")

        self.memory_data_queue = queue.Queue()
        self.monitoring_thread_should_stop = False

        self.memory_monitor_thread = threading.Thread(target=self._monitor_memory)
        self.memory_monitor_thread.daemon = True
        self.memory_monitor_thread.start()
        self.stop_logging_atexit_fn = lambda: self.stop_logging()
        atexit.register(self.stop_logging_atexit_fn)

        self.monitoring_in_progress = True

        return self

    def stop_logging(self):
        self.monitoring_thread_should_stop = True
        self.monitoring_in_progress = False
        self.memory_monitor_thread.join()
        self.log_memory_usage()
        atexit.unregister(self.stop_logging_atexit_fn)

    def log_memory_usage(self):
        memory_usage_data = list(self.memory_data_queue.queue)
        time_data, memory_data = tuple(zip(*memory_usage_data))
        time_data = subtract_first_element(list(time_data))
        if self.start_memory_from_zero:
            memory_data = subtract_first_element(list(memory_data))

        if not self.log_dir.exists():
            self.log_dir.mkdir(parents=True)

        # Save the memory usage data to a file
        with open(self.log_dir / f'memory_usage_{self.memory_type}.txt', 'w') as log_file:
            for timestamp, memory_usage in zip(time_data, memory_data):
                log_file.write(f"{timestamp} {memory_usage}\n")

            log_file.writelines([
                f"Total time: {time_data[-1] - time_data[0]}\n",
                f"Max memory: {max(memory_data)} (MB)"])

        fig = plt.figure(figsize=(10, 6))
        plt.plot(time_data, memory_data)
        plt.xlabel("Time (seconds)")
        plt.ylabel("Memory Usage (MB)")
        if self.plot_title:
            plt.title(f"Memory Usage vs. Time ({self.plot_title})")
        else:
            plt.title(f"Memory Usage vs. Time")
        plt.grid(True)
        plt.savefig(self.log_dir / f"memory_usage_{self.memory_type}.png")
        plt.close(fig)

    def _monitor_memory(self):
        while not self.monitoring_thread_should_stop:
            if self.memory_type == "rss":
                memory_usage = psutil.Process().memory_info().rss >> 20     # MB
            elif self.memory_type == "total-avail":
                memory_usage = (psutil.virtual_memory().total - psutil.virtual_memory().available) >> 20    # MB
            else:
                elem_filter = lambda it: "\x1b" not in it
                new_line_delimiter = "\x1b(B\x1b[m\x1b[39;49m"
                header_line = -3
                res_column = 5  # Resident Memory Size (KiB): The non-swapped physical memory a task is using.

                res = subprocess.run(f"top -n 1 -p {os.getpid()}".split(' '), capture_output=True, text=True)
                stdout, stderr = res.stdout, res.stderr
                lines = stdout.split(new_line_delimiter)
                if len(lines) < abs(header_line):
                    continue
                assert tuple(filter(elem_filter, lines[header_line].split()))[res_column] == "RES"
                line_elems = tuple(filter(elem_filter, lines[header_line + 1].split()))
                res_data = line_elems[res_column]
                if res_data.endswith('m') or res_data.endswith('g'):
                    float_value = float(res_data[:-1].replace(',', '.'))
                    bytes = float_value * 2**(30 if 'g' in res_data else 20)    # GiB or MiB to bytes
                    memory_usage = bytes / 10**6  # bytes to MB
                else:
                    memory_usage = (int(res_data) << 10) / 10**6  # kibibytes(KiB) to MB
            self.memory_data_queue.put((time.perf_counter(), int(memory_usage)))
            time.sleep(self.interval)


def subtract_first_element(data):
    for i in range(1, len(data)):
        data[i] = data[i] - data[0]
    data[0] = 0
    return data


if __name__ == '__main__':
    memory_logger = MemoryLogger(Path("./logs")).start_logging()
    import numpy as np
    from tqdm import tqdm
    a = []
    for i in tqdm(range(1000)):
        a.append(np.random.random((1000000)))
        memory_logger.log_memory_usage()
        time.sleep(0.1)
    