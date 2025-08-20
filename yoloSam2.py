import cv2
import matplotlib.pyplot as plt
import numpy as np
import sys

# Ultralytics 모듈 경로 추가 (필요 시)
sys.path.append("C:/Users/smpi9/Downloads/yolo_project/ultralytics")

from ultralytics import FastSAM

# 1. 이미지 경로와 모델 불러오기
image_path = "Dataset10/train/images/new_frame8_00503_0001color.png"
model = FastSAM("FastSAM-s.pt")  # 모델 경로

# 2. 이미지 로드 (BGR → RGB 변환)
image_bgr = cv2.imread(image_path)
image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

# 3. 클릭한 좌표 저장용 리스트
clicked_point = []

# 4. 클릭 이벤트 정의
def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point.append((x, y))
        print(f"📍 클릭한 좌표: ({x}, {y})")
        cv2.destroyAllWindows()

# 5. 이미지 띄우고 클릭 대기
cv2.imshow("Click a point for FastSAM segmentation", image_bgr)
cv2.setMouseCallback("Click a point for FastSAM segmentation", click_event)
cv2.waitKey(0)

# 6. 클릭 안 했으면 종료
if not clicked_point:
    print("❗ 포인트가 선택되지 않았습니다.")
    exit()

# 7. 모델 추론 (클릭한 좌표 사용)
x, y = clicked_point[0]
results = model(image_path, points=[(x, y)], labels=[1])

# 8. 마스크 추출 (첫 결과 기준)
mask = results[0].masks.data[0].cpu().numpy()

# 9. 마스크 이미지 크기에 맞게 resize
mask_resized = cv2.resize(mask, (image_rgb.shape[1], image_rgb.shape[0]))

# 10. 오버레이 생성
overlay = image_rgb.copy()
overlay[mask_resized > 0.5] = [255, 0, 0]  # 빨간색으로 표시

# 11. 시각화 출력
plt.figure(figsize=(10, 5))
plt.imshow(overlay)
plt.title(f"FastSAM Segmentation @ ({x}, {y})")
plt.axis('off')
plt.show()
