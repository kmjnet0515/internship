import cv2
import pyrealsense2 as rs
import numpy as np
import torch
import mediapipe as mp
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

# 내 커스텀 모델 경로 (전체 객체 탐지용)
custom_model_path = 'C:/Users/smpi9/Downloads/yolo_project/yolov5/runs/train/yolo_custom92/weights/best.pt'
custom_model = torch.hub.load('./yolov5', 'custom', path=custom_model_path, source='local', force_reload=True)
custom_model.eval()

# 일반 YOLOv5 사람 탐지용 모델 (사람만 인식)
person_model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
person_model.classes = [0]  # 사람 클래스만

device = 'cuda' if torch.cuda.is_available() else 'cpu'
custom_model = custom_model.to(device)
person_model = person_model.to(device)

print("모델 2개 로드 완료")

# MediaPipe Hands 초기화
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2,
                       min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

def init_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device('246322300435')
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    return pipeline, align

def Get_Frame(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth = aligned_frames.get_depth_frame()
    color = aligned_frames.get_color_frame()
    return np.asanyarray(color.get_data()), np.asanyarray(depth.get_data()), depth

print("시작")
pipeline, align = init_camera()

# 최초 프레임의 depth 저장
_, initial_depth_image, initial_depth_frame = Get_Frame(pipeline, align)
initial_depth_copy = initial_depth_image.copy()

while True:
    frame, depth_image, depth_frame = Get_Frame(pipeline, align)
    h, w = frame.shape[:2]

    # 1) 커스텀 모델용 이미지 전처리 (흑백->RGB 640x640)
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_custom = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    target_size = 1280
    if h < target_size:
        total_pad = target_size - h
        top_pad = total_pad // 2
        bottom_pad = total_pad - top_pad
    else:
        top_pad = bottom_pad = 0

    img_padded = cv2.copyMakeBorder(img_custom, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img_custom_resized = cv2.resize(img_padded, (1280, 1280))
    img_custom_tensor = torch.from_numpy(img_custom_resized).float() / 255.0
    img_custom_tensor = img_custom_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    # 2) 사람 모델용 이미지 전처리 (원본 컬러 1280x720를 YOLO 기본 입력 크기 640x640로 리사이즈)
    img_person = cv2.resize(frame, (640, 640))
    img_person_tensor = torch.from_numpy(img_person).float() / 255.0
    img_person_tensor = img_person_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    # 3) 추론
    with torch.no_grad():
        custom_results = custom_model(img_custom_tensor)[0]
        person_results = person_model(img_person_tensor)[0]

    # 원본 프레임 복사
    output_img = frame.copy()

    # --- 커스텀 모델: 객체 중심 깊이 차이 계산 후 차이 < 6m 인 경우만 표시 ---
    custom_detections = custom_results[custom_results[:,4] > 0.5]
    for det in custom_detections:
        x_center, y_center, width, height, conf, *class_probs = det.cpu().numpy()

        # 커스텀 모델 좌표는 640x640 기준 → 원본 1280x720 크기에 맞게 보정
        x_scale = w / 640
        y_scale = h / 640
        x_center_orig = int(x_center * x_scale)
        y_center_orig = int(y_center * y_scale)

        x1 = int((x_center - width / 2) * x_scale)
        y1 = int((y_center - height / 2) * y_scale)
        x2 = int((x_center + width / 2) * x_scale)
        y2 = int((y_center + height / 2) * y_scale)

        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w-1, x2)
        y2 = min(h-1, y2)

        # 현재 깊이값 및 초기 깊이값과 차이
        x_depth = np.clip(x_center_orig, 0, depth_image.shape[1] - 1)
        y_depth = np.clip(y_center_orig, 0, depth_image.shape[0] - 1)
        current_depth_val = depth_image[y_depth, x_depth]
        initial_depth_val = initial_depth_copy[y_depth, x_depth]
        current_dist = current_depth_val * 0.001
        initial_dist = initial_depth_val * 0.001
        depth_diff = current_dist - initial_dist

        if abs(depth_diff) < 6:
            label = f"{custom_model.names[int(cls)]} {conf:.2f} Dist:{current_dist:.2f}m Dif:{depth_diff:.2f}m"
            cv2.rectangle(output_img, (x1, y1), (x2, y2), (0,255,0), 2)
            cv2.putText(output_img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    # --- 사람 모델: 사람 bbox만 필터링 및 원본 좌표 변환 ---
    person_detections = person_results[person_results[:,4] > 0.5]
    for det in person_detections:
        arr = det.cpu().numpy()
        x1, y1, x2, y2, conf = arr[:5]
        class_probs = arr[5:]
        cls = int(np.argmax(class_probs))

        x1 = int(x1 * (w/640))
        y1 = int(y1 * (h/640))
        x2 = int(x2 * (w/640))
        y2 = int(y2 * (h/640))

        x1_clip = max(0, x1)
        y1_clip = max(0, y1)
        x2_clip = min(w-1, x2)
        y2_clip = min(h-1, y2)

        # 유효 좌표 검사
        if x2_clip <= x1_clip or y2_clip <= y1_clip:
            continue

        person_roi = output_img[y1_clip:y2_clip, x1_clip:x2_clip]
        person_roi_rgb = cv2.cvtColor(person_roi, cv2.COLOR_BGR2RGB)

        # MediaPipe Hands 처리
        results_hands = hands.process(person_roi_rgb)

        if results_hands.multi_hand_landmarks:
            for hand_landmarks in results_hands.multi_hand_landmarks:
                for id, lm in enumerate(hand_landmarks.landmark):
                    px = int(lm.x * (x2_clip - x1_clip))
                    py = int(lm.y * (y2_clip - y1_clip))

                    px_orig = x1_clip + px
                    py_orig = y1_clip + py

                    px_depth = np.clip(px_orig, 0, depth_image.shape[1] - 1)
                    py_depth = np.clip(py_orig, 0, depth_image.shape[0] - 1)

                    current_depth_val = depth_image[py_depth, px_depth]
                    initial_depth_val = initial_depth_copy[py_depth, px_depth]
                    current_dist = current_depth_val * 0.001
                    initial_dist = initial_depth_val * 0.001
                    depth_diff = current_dist - initial_dist

                    # 깊이 차이 화면 표시
                    cv2.circle(output_img, (px_orig, py_orig), 5, (0, 0, 255), -1)
                    cv2.putText(output_img, f"{current_dist:.2f}m", (px_orig+5, py_orig-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

                # MediaPipe 랜드마크 그리기 (ROI에)
                mp_drawing.draw_landmarks(person_roi, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.rectangle(output_img, (x1_clip, y1_clip), (x2_clip, y2_clip), (255, 0, 0), 2)
        cv2.putText(output_img, 'person', (x1_clip, y1_clip - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,0,0), 2)

    cv2.imshow('Custom & Person Detection + Hands + Depth', output_img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

pipeline.stop()
cv2.destroyAllWindows()
