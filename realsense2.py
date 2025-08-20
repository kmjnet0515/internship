import cv2
import pyrealsense2 as rs
import numpy as np
import sys
import os
import torch
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath
import mediapipe as mp
import time
from deep_sort_realtime.deepsort_tracker import DeepSort
import sys
from ultralytics import YOLO

sys.path.append("C:/Users/smpi9/Downloads/yolo_project/yolov5")
#sys.path.append("C:/Users/smpi9/Downloads/yolo_project/ultralytics")
# ✅ yolov5 디렉토리 안의 models.common에서 DetectMultiBackend 가져오기
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import non_max_suppression
model = YOLO("yolov11s.pt")  # 또는 "best.pt"
      
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
device = torch.device('cpu')
# ✅ DetectMultiBackend로 모델 로드

print(type(model))
print(model.names)
print(len(model.names))
print(torch.__version__)
model2 = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model2.classes = [0]  # 사람만 감지

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                    enable_segmentation=False, min_detection_confidence=0.8)
mp_drawing = mp.solutions.drawing_utils
print("모델 로드 완료")
th = 0.1

# ✅ DeepSORT 초기화 (물체만)
tracker = DeepSort(max_age=20)

def init_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    ctx = rs.context()
    devices = ctx.query_devices()
    if not devices:
        raise RuntimeError("❌ 연결된 RealSense 장치가 없습니다.")

    serial = devices[1].get_info(rs.camera_info.serial_number)
    print(f"✅ 연결된 장치 시리얼: {serial}")
    config.enable_device(serial)
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

# 벽 상태 설정 메서드
def set_touch_wall(track, value: bool):
    track.touch_wall = value

# 벽 여부 확인 메서드
def is_touch_wall(track):
    return getattr(track, "touch_wall", False)
objectList = []

print("시작")
pipeline, align = init_camera()
start = time.time()
while time.time() - start < 3:
    print(f"{(3 - (time.time() - start)):.2f}초 남음")
    frame, depth_image, depth_frame = Get_Frame(pipeline, align)
    current_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    cv2.imshow('YOLOv5 Real-time', frame)
    cv2.imshow('Current Depth', current_depth_colormap)
    cv2.waitKey(1)

_, initial_depth_image, initial_depth_frame = Get_Frame(pipeline, align)
initial_depth_copy = initial_depth_image.copy()
k = 0
while True:
    k += 1
    frame, depth_image, depth_frame = Get_Frame(pipeline, align)
    h, w = frame.shape[:2]

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model2(img_rgb)

    boxes = results.xyxy[0].cpu().numpy()

    target_size = 1280
    if h < target_size:
        total_pad = target_size - h
        top_pad = total_pad // 2
        bottom_pad = total_pad - top_pad
    else:
        top_pad = bottom_pad = 0

    img_padded = cv2.copyMakeBorder(frame, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img_resized = cv2.resize(img_padded, (1280, 1280))
    img_copy = img_resized.copy()
    gray_frame = cv2.cvtColor(img_resized, cv2.COLOR_BGR2GRAY)
    img = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    img_resized = img
    for box in boxes:
        x1, y1, x2, y2, conf, cls = box
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        person_img = frame[y1:y2, x1:x2].copy()
        if person_img.size == 0:
            continue

        img_rgb_person = cv2.cvtColor(person_img, cv2.COLOR_BGR2RGB)
        results_pose = pose.process(img_rgb_person)
        if results_pose.pose_landmarks:
            pose_landmarks = results_pose.pose_landmarks.landmark
            left_hand_indices = [13, 15]
            right_hand_indices = [14, 16]
            hand_indices = left_hand_indices + right_hand_indices
            for idx in hand_indices:
                lm = pose_landmarks[idx]
                x_pixel = int(lm.x * (x2 - x1))
                y_pixel = int(lm.y * (y2 - y1))
                x_depth = x1 + x_pixel
                y_depth = y1 + y_pixel
                x_depth = np.clip(x_depth, 0, 1279)
                y_depth = np.clip(y_depth, 0, 719)
                current_depth_value = depth_image[y_depth, x_depth]
                current_distance = current_depth_value * 0.001
                initial_depth_value = initial_depth_copy[y_depth, x_depth]
                initial_distance = initial_depth_value * 0.001
                depth_diff = initial_distance - current_distance
                color = (0, 255, 255) if depth_diff < th else (255, 0, 0)
                depth_label = f"{depth_diff:.2f}"
                cv2.putText(person_img, depth_label, (x_pixel, y_pixel+20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                cv2.circle(person_img, (x_pixel, y_pixel), 5, color, -1)
            img_copy[y1+280:y2+280, x1:x2] = person_img
            cv2.rectangle(img_copy, (x1, y1+280), (x2, y2+280), (255, 0, 0), 2)

    img_tensor = torch.from_numpy(img_resized).float() / 255.0
    img_tensor = img_tensor.permute(2, 0, 1).unsqueeze(0).to(device)

    pred = model(img_tensor)  # raw tensor
    for result in results:
        boxes = result.boxes  # Boxes object

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            print(f"Box: ({x1:.2f}, {y1:.2f}) → ({x2:.2f}, {y2:.2f}), conf: {confidence:.2f}, class: {class_id}")
            cv2.rectangle(img_copy, (x1,y1), (x2,y2), color, 2)
            cv2.putText(img_copy, model.name[class_id], (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
    '''pred = non_max_suppression(pred, conf_thres=0.5, iou_thres=0.4)

    # pred는 리스트, pred[0]에 (num_boxes, 6) 형태 [x1,y1,x2,y2,conf,class]
    if pred[0] is not None:
        detected_objects_coordinates = pred[0].cpu().numpy()
    else:
        detected_objects_coordinates = []

    detections = []
    for det in detected_objects_coordinates:
        x1, y1, x2, y2, conf, cls = det.tolist()
        print(det.tolist())
        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)
        w = x2 - x1
        h = y2 - y1
        label = f"{model.names[int(cls)]} {conf:.2f}"

        x1 = max(0, x1)
        y1 = max(0, y1)
        w = max(1, min(w, frame.shape[1] - x1))
        h = max(1, min(h, frame.shape[0] - y1))
        if w <= 1 or h <= 1:
            continue
        detections.append(([x1, y1, w, h], conf, model.names[int(cls)]))


    print(detections)
    # 2) DeepSORT 업데이트
    tracks = tracker.update_tracks(detections, frame=frame)
    # 3) 트랙별로 상태 갱신 및 그리기
    for track in tracks:
        if not track.is_confirmed():
            continue

        if not hasattr(track, "touch_wall"):
            track.touch_wall = False
 
        try:
            tid = int(track.track_id)
        except ValueError:
            tid = None
        if track.time_since_update < 1:
            ltrb = track.to_ltrb()
            tx1, ty1, tx2, ty2 = map(int, ltrb)
            x_center = int((tx1 + tx2)/2)
            y_center = int((ty1 + ty2)/2)-280

            x_depth = np.clip(x_center, 0, 1279)
            y_depth = np.clip(y_center, 0, 719)

            current_depth_value = depth_image[y_depth, x_depth]
            current_distance = current_depth_value * 0.001
            initial_depth_value = initial_depth_copy[y_depth, x_depth]
            initial_distance = initial_depth_value * 0.001
            depth_diff = initial_distance - current_distance
            label_text = f"{depth_diff:.2f}"

            color = (0, 255, 255) if depth_diff < th else (0, 0, 255) if track.touch_wall else (255,0,0)
            if tid is not None:
                if depth_diff < th:
                    track.touch_wall = True
                cv2.putText(img_copy, label_text, (tx1, ty1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
           
            
            
            label_text = f"ID:{track.track_id} {'Touch Wall' if track.touch_wall else ''}"

            cv2.rectangle(img_copy, (tx1, ty1), (tx2, ty2), color, 2)
            cv2.putText(img_copy, label_text, (tx1, ty1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 1)
    '''     

    # (원하는 경우 거리 표시 등도 추가 가능)


    img_copy = img_copy[280:1000, :]
    background_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(initial_depth_image, alpha=0.03), cv2.COLORMAP_JET)
    current_depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    combined_img = np.vstack((img_copy, current_depth_colormap))
    cv2.imshow('YOLOv5 Real-time Detection (Grayscale + Depth)', img_copy)
    cv2.imwrite(f'videos/newFrame{k}.png', combined_img)
    cv2.imshow('Background Depth', background_depth_colormap)
    cv2.imshow('Current Depth', current_depth_colormap)

    if cv2.waitKey(1) & 0xFF == 27:
        break

pipeline.stop()
cv2.destroyAllWindows()
