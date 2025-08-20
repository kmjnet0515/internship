import sys
import os
import torch
import cv2
import numpy as np
import pyrealsense2 as rs
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath

model_path = 'C:/Users/smpi9/Downloads/yolo_project/yolov5/runs/train/yolo_custom7/weights/best.pt'
model = torch.hub.load('./yolov5', 'custom', path=model_path, source='local', force_reload=True)
model.eval()

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(device)

print("모델 로드 완료")



target_size = 1280
def init_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device('339522300522')
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
    return np.asanyarray(color.get_data()), np.asanyarray(depth.get_data())
pipeline, align = init_camera()
try:
    while True:
        
        color_frame, Depth=Get_Frame(pipeline, align)


        # numpy 배열로 변환
        frame =cv2.cvtColor(color_frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]

        # 그레이스케일 변환
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

        # 상하 여백 패딩 계산
        if h < target_size:
            total_pad = target_size - h
            top_pad = total_pad // 2
            bottom_pad = total_pad - top_pad
        else:
            top_pad = bottom_pad = 0

        left_pad = right_pad = 0  # 좌우는 이미 1280이므로 패딩 없음

        img_padded = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        img_resized = cv2.resize(img_padded, (target_size, target_size))
        img_copy = img_resized.copy()

        # Tensor 변환 및 정규화
        img_tensor = torch.from_numpy(img_resized).float()
        img_tensor /= 255.0
        img_tensor = img_tensor.permute(2, 0, 1)  # C,H,W
        img_tensor = img_tensor.unsqueeze(0).to(device)

        # 추론
        results = model(img_tensor)

        detected_objects = results[0][results[0][:, 4] > 0.1]

        for det in detected_objects:
            x_center, y_center, width, height, conf, *class_probs = det
            x1 = int((x_center - width / 2))
            y1 = int((y_center - height / 2))
            x2 = int((x_center + width / 2))
            y2 = int((y_center + height / 2))

            cls = torch.argmax(torch.tensor(class_probs))
            label = f"{model.names[int(cls)]} {conf:.2f}"

            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow('YOLOv5 Real-time Detection (Grayscale)', img_copy)
        cv2.imshow('Original Frame', frame)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC 키로 종료
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
