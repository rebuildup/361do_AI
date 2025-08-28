#!/usr/bin/env python3
import psutil
import time
import json
from datetime import datetime

def monitor_system():
    while True:
        stats = {
            'timestamp': datetime.now().isoformat(),
            'cpu_percent': psutil.cpu_percent(),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('.').percent,
            'process_count': len(psutil.pids())
        }
        
        with open('system_monitor.log', 'a') as f:
            f.write(json.dumps(stats) + '\n')
        
        time.sleep(60)  # 1•ªŠÔŠu

if __name__ == "__main__":
    monitor_system()
