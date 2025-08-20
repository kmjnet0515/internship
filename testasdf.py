from ultralytics import YOLO
import cv2
from random import randint
model = YOLO("yolov8m-pose.pt")
model.eval()

img = cv2.imread("yolov5/data/images/bus.jpg")
results = model.predict(img)

joint_list = []  # 팔꿈치와 손목 좌표 저장 리스트

for r in results:
    keypoints = r.keypoints
    kpts = keypoints.xy.cpu().numpy()   # shape: (num_people, 17, 2)
    confs = keypoints.conf.cpu().numpy()  # shape: (num_people, 17)

    for person_idx in range(len(kpts)):
        person_kpts = kpts[person_idx]
        person_confs = confs[person_idx]
        a = randint(0,255)
        b = randint(0,255)
        c = randint(0,255)
        # 필요한 관절 인덱스 (팔꿈치: 7,8 / 손목: 9,10)
        for idx in [7, 8, 9, 10]:
            if idx < len(person_confs) and person_confs[idx] >= 0.5:
                x, y = map(int, person_kpts[idx])
                joint_list.append((x, y))
                # 시각화: 파란색 원
                cv2.circle(img, (x, y), radius=5, color=(c, c, c), thickness=-1)

# 디버깅 출력
print("📍 팔꿈치 + 손목 좌표:")
for i, pt in enumerate(joint_list):
    print(f"{i+1}: {pt}")

# 이미지 보기
cv2.imshow("elbows and wrists", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
