import os
import time
while True:
    os.system("netstat -ano | findstr :8765")
    time.sleep(1)
