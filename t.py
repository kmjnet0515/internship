from ultralytics import YOLO
import sys

sys.path.append("C:/Users/smpi9/Downloads/yolo_project/ultralytics")
# 모델 로드
model = YOLO('yolo11m.pt')

# 학습 (직사각형 이미지 크기 지정, 예: 1280x720)
model.train(
    data='/usr/src/app/data.yaml',
    epochs=20,
    imgsz=(720, 1280),  # (height, width) 튜플로 지정 — 직사각형 가능
    batch=4,
    name='yolo_7nc4_rect',
    workers=4,
    device='cuda'  # GPU 사용 시
)