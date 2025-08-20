import torch
import cv2
import os

# YOLOv5s 모델 로드 (사전 학습된 모델, 사람 클래스 포함)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.4  # confidence threshold
model.classes = [0]  # only detect person

# 경로 설정
img_dir = 'Dataset6/val/images'
label_dir = 'Dataset6/val/labels'

# 이미지 파일 목록
img_files = [f for f in os.listdir(img_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

for img_file in img_files:
    # 이미지 경로
    img_path = os.path.join(img_dir, img_file)
    img = cv2.imread(img_path)
    h, w = img.shape[:2]

    # 추론
    results = model(img)
    detections = results.xyxy[0].cpu().numpy()

    # 사람만 박스로 그리기
    temp_img = img.copy()
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        cv2.rectangle(temp_img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(temp_img, f"person {conf:.2f}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 이미지 보여주기
    cv2.imshow("Person Detection", temp_img)
    key = cv2.waitKey(1) & 0xFF

    # 라벨 파일 경로
    label_path = os.path.join(label_dir, os.path.splitext(img_file)[0] + '.txt')

    # 기존 라벨 읽기 (없으면 빈 리스트)
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            existing_labels = f.read().splitlines()
    else:
        existing_labels = []
    if not existing_labels:
        input("라벨이 없음")
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        # YOLO 포맷 변환
        x_center = ((x1 + x2) / 2) / w
        y_center = ((y1 + y2) / 2) / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h
        yolo_label = f"0 {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
        existing_labels.append(yolo_label)

    with open(label_path, 'w') as f:
        f.write('\n'.join(existing_labels))
    print(f"[✔] 저장됨 (사람 포함): {label_path}")



cv2.destroyAllWindows()
