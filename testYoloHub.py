import sys
sys.path.append("C:/Users/smpi9/Downloads/yolo_project/ultralytics")

from ultralytics import YOLO, checks, hub
checks()

hub.login('cb485e88b36fedab302c1024edda43a264dd1bb70e')

model = YOLO('https://hub.ultralytics.com/models/Y5lYJVTHn8n56b3mT0xg')
results = model.train()