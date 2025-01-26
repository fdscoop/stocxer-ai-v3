# 8. Add memory monitoring
# monitoring.py
import psutil
import os

def get_memory_usage():
    process = psutil.Process(os.getpid())
    return {
        'memory_percent': process.memory_percent(),
        'memory_info': process.memory_info()._asdict()
    }