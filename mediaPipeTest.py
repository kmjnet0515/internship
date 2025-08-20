import cv2
import torch
import mediapipe as mp
import numpy as np

# YOLOv5 모델 로드
model2 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model2.classes = [0]  # 사람만 감지 (COCO 클래스 ID: 0 = person)

# MediaPipe Pose 초기화
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    enable_segmentation=False, min_detection_confidence=0.5)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv5에 맞게 BGR -> RGB
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model2(img_rgb)

    # 감지된 사람들 바운딩 박스 가져오기
    boxes = results.xyxy[0].cpu().numpy()

    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])

        # Crop된 사람 이미지 가져오기
        person_img = frame[y1:y2, x1:x2].copy()

        if person_img.size == 0:
            continue

        # MediaPipe Pose 적용
        img_rgb_person = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(img_rgb_person)

        if results_pose.pose_landmarks:
            mp_drawing.draw_landmarks(
                person_img,
                results_pose.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
            )

        # 원래 프레임에 다시 붙여넣기
        frame[y1:y2, x1:x2] = person_img

        # YOLO 박스도 표시
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

    cv2.imshow('YOLOv5 + MediaPipe Pose', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

pose.close()
cap.release()
cv2.destroyAllWindows()
