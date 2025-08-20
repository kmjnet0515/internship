import os
import cv2

# 색상 및 클래스 이름 정의
LABEL_COLORS = [
    (0, 0, 255), (0, 165, 255), (0, 255, 255),
    (0, 255, 0), (255, 0, 0), (255, 0, 255),
    (128, 0, 128),
]
class_names = ['Anchor', 'SeaReaf', 'SeaSquirt', 'WaterDrop', 'WhirlPool', 'FishingHook', 'WhiteMullet']

# 이미지 및 라벨 폴더 분리
image_folder = 'Dataset7/train/images'
label_folder = 'Dataset7/train/labels'

# 이미지 파일만 선택
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

pre = []
pre2 = []

for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    label_file = os.path.splitext(image_file)[0] + '.txt'
    label_path = os.path.join(label_folder, label_file)

    if not os.path.exists(label_path):
        continue

    image = cv2.imread(image_path)
    if image is None:
        print(f"이미지 로드 실패: {image_path}")
        continue
    h, w = image.shape[:2]

    with open(label_path, 'r') as f:
        lines = f.readlines()

    # 이전 라벨 표시
    if pre:
        pre2 = []
        for p in pre:
            cv2.rectangle(image, (p[0]-5, p[1]-5), (p[2]+5, p[3]+5), p[4], 2)
            pre2.append(p[:])
        pre.clear()

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cls = int(parts[0])
        x_center, y_center, box_w, box_h = map(float, parts[1:5])
        x_center *= w
        y_center *= h
        box_w *= w
        box_h *= h

        x1 = int(x_center - box_w / 2)
        y1 = int(y_center - box_h / 2)
        x2 = int(x_center + box_w / 2)
        y2 = int(y_center + box_h / 2)

        color = LABEL_COLORS[cls % len(LABEL_COLORS)]
        label = class_names[cls]

        for p in pre2:
            if abs(p[0] - x1) <= 20 and abs(p[1] - y1) <= 20 and p[-1] != label:
                print("pre :", p)
                print(f"cur : ({x1}, {y1}, {x2}, {y2}, {color}, {label})")
                cv2.rectangle(image, (x1-10, y1-10), (x2+10, y2+10), (0, 0, 0), 5)
                cv2.imshow("detect", image)
                cv2.waitKey(1)
                input(f"⚠ 다른 라벨 감지됨. {image_path}, {label_path}")

        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(image, label, (x1, max(y1 - 5, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        pre.append((x1, y1, x2, y2, color, label))

    cv2.imshow("Labeled", image)
    cv2.waitKey(1)

cv2.destroyAllWindows()
