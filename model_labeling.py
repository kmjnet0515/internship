import os
import cv2
import torch
import sys
import numpy as np
import pathlib
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath
sys.path.append("C:/Users/smpi9/Downloads/yolo_project/ultralytics")
from ultralytics import YOLO
# 모델 로드
model = YOLO('C:/Users/smpi9/Downloads/yolo_project/ultralytics/runs/detect/yolo_7nc3/weights/best.pt')


# 디바이스 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(torch.device('cpu'))

# 이미지가 저장된 폴더 경로
image_folder_path = 'new_frame4'
output_folder_path = 'new_saved_frame41'
output_color_folder_path = 'new_saved_frame42'
os.makedirs(output_folder_path, exist_ok=True)
os.makedirs(output_color_folder_path, exist_ok=True)
# 이미지 파일 목록
image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# YOLO 클래스 이름
class_names = ['Anchor','SeaReaf', 'SeaSquirt', 'WaterDrop', 'WhirlPool',"FishingHook",'WhiteMullet']
LABEL_COLORS = [
    (0, 0, 255),      # 빨강
    (0, 165, 255),    # 주황
    (0, 255, 255),    # 노랑
    (0, 255, 0),      # 초록
    (255, 0, 0),      # 파랑
    (255, 0, 255),    # 남보라
    (128, 0, 128),    # 보라
    (0,0,0),
]
def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"이미지 로드 실패: {image_path}")
        return None

    h, w = frame.shape[:2]
    print(f"Processing image: {image_path}, size: {h}x{w}")
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 이미지를 RGB로 변환 (모델이 RGB로 입력을 받으므로)
    img = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    
    
    #frame_copy = frame.copy()
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    #img = frame
    target_size = 1280

    # padding 계산
    if h < target_size:
        total_pad = target_size - h
        top_pad = total_pad // 2
        bottom_pad = total_pad - top_pad
    else:
        top_pad = bottom_pad = 0

    # 좌우 padding은 필요 없음 (이미 1280)
    left_pad = right_pad = 0

    # padding 적용
    img_padded = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    # 이미지 크기 리사이즈
    img_resized = cv2.resize(img_padded, (1280,1280))
    
    # 이미지를 Tensor로 변환하고, [H, W, C] -> [C, H, W] 순서로 변경
    img = torch.from_numpy(img_resized).float()
    img /= 255.0  # Normalize to [0, 1]
    img = img.permute(2, 0, 1)  # [C, H, W]로 변환

    # 배치 차원을 추가하고 device로 전송
    img = img.unsqueeze(0).to(device)

    # 추론 (모델에 이미지를 넣어 결과를 얻음)
    results = model(img)

    ''' # 결과에서 bounding boxes, labels, and scores 추출
    detected_objects_coordinates = results[0][results[0][:, 4] > 0.3]  # confidence threshold 0.5

    # Non-Maximum Suppression (NMS) 적용
    det_boxes = detected_objects_coordinates[:, :4]  # (x1, y1, x2, y2)
    scores = detected_objects_coordinates[:, 4]  # confidence scores
    classes = detected_objects_coordinates[:, 5]  # predicted classes
    indices = cv2.dnn.NMSBoxes(det_boxes.cpu().numpy(), scores.cpu().numpy(), score_threshold=0.5, nms_threshold=0.4)

    # NMS가 반환한 인덱스를 이용하여 중복 제거된 객체들만 반환
    nms_detected_objects = []
    if len(indices) > 0:
        for i in indices.flatten():
            nms_detected_objects.append(detected_objects_coordinates[i])
    
    # 패딩 정보를 process_image 함수에서 반환
    return nms_detected_objects, img_resized, w, h, top_pad, bottom_pad, frame
    '''
    return results, img, w, h, top_pad, bottom_pad, frame



def save_labels(image_name, detected_objects, output_path, w, h, top_pad, bottom_pad):
    os.makedirs(output_path, exist_ok=True)

    label_file_path = os.path.join(output_path, f"{os.path.splitext(image_name)[0]}.txt")
    with open(label_file_path, 'w') as f:
        for result in detected_objects:
            boxes = result.boxes  # Boxes object

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                y1 -=280
                y2 -=280
                confidence = box.conf.item()
                cls = int(box.cls.item())
                width = x2-x1
                height = y2 - y1
                x_center, y_center = x1 + width/2, y1 + height/2
                
                # 패딩된 이미지 기준이므로 먼저 원본 기준으로 변환
                y_center_unpad = y_center - top_pad
                # 패딩된 이미지에서 width, height는 동일, x축에는 패딩 없음

                # 원본 이미지의 크기로 나누어 정규화
                x_center_norm = x_center / w
                y_center_norm = y_center_unpad / h
                width_norm = width / w
                height_norm = height / h

                f.write(f"{cls} {x_center_norm:.6f} {y_center_norm:.6f} {width_norm:.6f} {height_norm:.6f}\n")
    print(f"Saved label to {label_file_path}")




# 이미지 처리
# 이미지 처리

for image_file in image_files:
    image_path = os.path.join(image_folder_path, image_file)
    '''if image_path[image_path.find('frame')+5:image_path.find('frame')+8] == '220':
           continue'''
    detected_objects, frame, w, h, top_pad, bottom_pad, original_image = process_image(image_path)
    #frame = frame[280:1000, :]
    frame_copy = frame.copy()
    
    if detected_objects is not None:
        data = []
        for result in detected_objects:
            boxes = result.boxes  # Boxes object

            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                x1, y1, x2, y2 = map(int, (x1,y1,x2,y2))
                
                confidence = box.conf.item()
                class_id = int(box.cls.item())
                print(f"Box: ({x1:.2f}, {y1:.2f}) → ({x2:.2f}, {y2:.2f}), conf: {confidence:.2f}, class: {class_id}")
                data.append(class_names[class_id])
                cv2.rectangle(frame, (x1,y1-280), (x2,y2-280), (0,0,255), 2)
                cv2.putText(frame, model.names[class_id], (x1, y1-280-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255), 1)
        '''
        for det in detected_objects:
            x_center, y_center, width, height, conf, *class_probs = det
            # 좌표 변환
            y_center_unpad = y_center - top_pad
            x1 = int(x_center - width / 2)
            y1 = int(y_center_unpad - height / 2)
            x2 = int(x_center + width / 2)
            y2 = int(y_center_unpad + height / 2)
            index = int(torch.argmax(torch.tensor(class_probs)).item())
            color = LABEL_COLORS[index % len(LABEL_COLORS)]
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            label = f"{class_names[index]} {conf:.2f}"

            data.append(class_names[index])
            cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)'''
        cv2.imshow('Detection Result', frame)
        cv2.waitKey(1)
        if not (data.count('Anchor') == 2 and data.count('SeaReaf') == 2 and data.count("SeaSquirt") == 2 and data.count('WaterDrop') == 2 and data.count("WhirlPool") == 2 and data.count('FishingHook') == 2 and data.count('WhiteMullet') == 2):
            continue

        

        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == ord('a'):
                cv2.imwrite(os.path.join(output_folder_path, image_file), frame_copy)
                cv2.imwrite(os.path.join(output_color_folder_path, image_file), original_image)
                save_labels(image_file, detected_objects, output_folder_path, w, h, top_pad, bottom_pad)
                print(f"Saved image: {image_file}")
                break
            elif key == ord('b'):
                break
        

cv2.destroyAllWindows()
