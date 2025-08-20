import sys
import os
import torch
import cv2
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

model_path = 'C:/Users/smpi9/Downloads/yolo_project/yolov5/runs/train/yolo5_custom7/weights/best.pt'

# attempt_load 함수로 모델 로드 (autoshape 제거)
model = torch.hub.load('./yolov5', 'custom', path=model_path, source='local', force_reload=True)
model.eval()
# 디바이스 설정 (CUDA가 가능하면 CUDA 사용, 아니면 CPU 사용)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

print("모델 로드 완료")

# 카메라 열기 (장치 번호 0)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("카메라 열기 실패")
    exit()
for i in range(100):
    try:
        frame = cv2.imread(f"DataSet/train/images/SeaReaf_{i}.jpg")
    except:
        continue
    h, w = frame.shape[:2]
    print(h,w)
    # 이미지를 흑백으로 변환
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 이미지를 RGB로 변환 (모델이 RGB로 입력을 받으므로)
    img = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # 이미지 크기 리사이즈
    img_resized = cv2.resize(img, (640, 640))
    img_copy = img_resized.copy()

    # 이미지를 Tensor로 변환하고, [H, W, C] -> [C, H, W] 순서로 변경
    img = torch.from_numpy(img_resized).float()
    img /= 255.0  # Normalize to [0, 1]
    img = img.permute(2, 0, 1)  # [C, H, W]로 변환

    # 배치 차원을 추가하고 device로 전송
    img = img.unsqueeze(0).to(device)

    # 추론 (모델에 이미지를 넣어 결과를 얻음)
    results = model(img)  # 모델에 이미지 전달

    # 결과에서 bounding boxes, labels, and scores 추출
    print(results[0][:,4])
    detected_objects_coordinates = results[0][results[0][:,4] > 0.3]

    # 좌표 출력
    print("Detected Objects Coordinates:")
    for q in detected_objects_coordinates:
        print(q[4])
    print()
    #print(detected_objects_coordinates[0][0])

    # 이미지에 바운딩 박스 그리기
    for det in detected_objects_coordinates:  # det = [x_center, y_center, width, height, conf, class1_conf, class2_conf, ..., classN_conf]
        x_center, y_center, width, height, conf, *class_probs = det

        # 바운딩 박스의 좌상단(x1, y1)과 우하단(x2, y2) 좌표 계산
        x1 = int((x_center - width / 2))  # 왼쪽 상단 x 좌표
        y1 = int((y_center - height / 2))  # 왼쪽 상단 y 좌표
        x2 = int((x_center + width / 2))  # 오른쪽 하단 x 좌표
        y2 = int((y_center + height / 2))  # 오른쪽 하단 y 좌표

        # 가장 높은 확률을 가진 클래스 인덱스
        cls = torch.argmax(torch.tensor(class_probs))  # 가장 큰 확률의 클래스를 선택

        # 라벨을 그려주기: 클래스 이름과 신뢰도를 포함
        label = f"{model.names[int(cls)]} {conf:.2f}"

        # 바운딩 박스 그리기
        cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)  # 바운딩 박스 색상: 초록
        cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)  # 라벨




    # 결과 출력
    cv2.imshow('YOLOv5 Real-time Detection (Grayscale)', img_copy)

    cv2.waitKey(100)



cv2.destroyAllWindows()
