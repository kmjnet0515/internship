import os
import cv2
import torch
import numpy as np
import pathlib
import sys
from pathlib import Path

pathlib.PosixPath = pathlib.WindowsPath
sys.path.append("C:/Users/smpi9/Downloads/yolo_project/ultralytics")

from ultralytics import YOLO

# 모델 로드
model = YOLO('C:/Users/smpi9/Downloads/yolo_project/ultralytics/runs/detect/best3(99).pt')
model2 = YOLO('C:/Users/smpi9/Downloads/yolo_project/ultralytics/runs/detect/best11.pt')

# 디바이스 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(torch.device('cuda'))
model2 = model2.to(torch.device('cuda'))

# 폴더 설정
image_folder_path = 'new_frame15'
output_folder_path = 'new_saved_frame151'
output_color_folder_path = 'new_saved_frame153'
os.makedirs(output_folder_path, exist_ok=True)
os.makedirs(output_color_folder_path, exist_ok=True)
image_files = [f for f in os.listdir(image_folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

class_names = ['Anchor', 'SeaReaf', 'SeaSquirt', 'WaterDrop', 'WhirlPool',"FishingHook",'WhiteMullet','SubmarineVolcano', 'DeepSeaHotSpring', 'StaticStair', 'DynamicStair']

current_boxes = []  # [x1, y1, x2, y2, class_id]
selected_box_index = None

# 드래그 상태용 변수
drawing = False
start_pos = None
end_pos = None

def process_image(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Failed to load image: {image_path}")
        return None, None, None, None, None, None, None, None
    color = frame.copy()
    h, w = frame.shape[:2]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # --- model1 resize (1280) ---
    target_size1 = 1280
    if h < target_size1:
        total_pad = target_size1 - h
        top_pad = total_pad // 2
        bottom_pad = total_pad - top_pad
    else:
        top_pad = bottom_pad = 0

    img_padded1 = cv2.copyMakeBorder(img, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img_resized1 = cv2.resize(img_padded1, (1280, 1280))
    tensor_img1 = torch.from_numpy(img_resized1).float() / 255.0
    tensor_img1 = tensor_img1.permute(2, 0, 1).unsqueeze(0).to(device)

    # --- model2 resize (640) ---
    target_size2 = 640
    img_resized2 = cv2.resize(img_padded1, (target_size2, target_size2))
    tensor_img2 = torch.from_numpy(img_resized2).float() / 255.0
    tensor_img2 = tensor_img2.permute(2, 0, 1).unsqueeze(0).to(device)

    # model1 inference
    results1 = model(tensor_img1)
    torch.cuda.empty_cache()

    # model2 inference, 낮은 confidence
    results2 = model2(tensor_img2)

    return results1, results2, img_resized1, w, h, top_pad, bottom_pad, color, img2

def save_labels(image_name, boxes, output_path, w, h, top_pad, bottom_pad):
    label_file_path = os.path.join(output_path, f"{os.path.splitext(image_name)[0]}.txt")
    with open(label_file_path, 'w') as f:
        for x1, y1, x2, y2, cls_id in boxes:
            xc = (x1 + x2) / 2 / w
            yc = (y1 + y2) / 2 / h
            ww = (x2 - x1) / w
            hh = (y2 - y1) / h
            f.write(f"{cls_id} {xc:.6f} {yc:.6f} {ww:.6f} {hh:.6f}\n")

def draw_boxes(frame, list1):
    color_list = []
    for i in range(len(class_names)):
        if i <= 6:  # 0~6번 클래스
            color_list.append((255,255,0) if list1.count(i) == 2 else (0,255,0))
        else:       # 7~10번 클래스
            color_list.append((255,255,0) if list1.count(i) == 1 else (0,255,0))

    for i, (x1, y1, x2, y2, cls_id) in enumerate(current_boxes):
        color = color_list[cls_id]
        if i == selected_box_index:
            color = (0, 0, 255)  # 선택된 박스 빨간색
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cls_id]}"
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def mouse_callback(event, x, y, flags, param):
    global current_boxes, selected_box_index, drawing, start_pos, end_pos
    if event == cv2.EVENT_LBUTTONDOWN:
        isEditing = False
        diff = 10
        for i, (x1,y1,x2,y2,_) in enumerate(current_boxes):
            if abs(x1-x) <= diff and abs(y1-y) <= diff:
                current_boxes[i][0] = x
                current_boxes[i][1] = y
                selected_box_index = i
                isEditing = True
            elif abs(x2-x) <= diff and abs(y2-y) <= diff:
                current_boxes[i][2] = x
                current_boxes[i][3] = y
                selected_box_index = i
                isEditing = True
            elif abs(x1-x) <= diff and abs(y2-y) <= diff:
                current_boxes[i][0] = x
                current_boxes[i][3] = y
                selected_box_index = i
                isEditing = True
            elif abs(x2-x) <= diff and abs(y1-y) <= diff:
                current_boxes[i][2] = x
                current_boxes[i][1] = y
                selected_box_index = i
                isEditing = True
        if not isEditing:
            drawing = True
            start_pos = (x, y)
            end_pos = (x, y)
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        end_pos = (x, y)
    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False
        end_pos = (x, y)
        if start_pos and end_pos:
            x1, x2 = sorted([start_pos[0], end_pos[0]])
            y1, y2 = sorted([start_pos[1], end_pos[1]])
            current_boxes.append([x1, y1, x2, y2, 8])
            selected_box_index = len(current_boxes)-1
        start_pos = end_pos = None
    elif event == cv2.EVENT_MBUTTONDOWN:
        for i, (x1, y1, x2, y2, _) in enumerate(current_boxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_box_index = i
                break
        else:
            selected_box_index = None
    elif event == cv2.EVENT_RBUTTONDOWN:
        for i, (x1, y1, x2, y2, _) in enumerate(current_boxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                del current_boxes[i]
                selected_box_index = None
                break
    elif event == cv2.EVENT_MOUSEWHEEL:
        if selected_box_index is not None:
            direction = 1 if flags > 0 else -1
            current_boxes[selected_box_index][4] = (current_boxes[selected_box_index][4] + direction) % len(class_names)

# --- Main Loop ---
cout = 0
en = False
for image_file in image_files:
    image_path = os.path.join(image_folder_path, image_file)
    if int(image_path[-9:-4]) <= 363:
        continue
    if en:
        if cout < 10:
            cout += 1
            continue
        else:
            cout = 0
            en = False

    detected1, detected2, frame, w, h, top_pad, bottom_pad, color_frame, gr = process_image(image_path)
    frame = frame[280:1000, :]
    current_boxes = []

    # --- model1 좌표 처리 ---
    for result in detected1:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls.item())
            y1 -= 280
            y2 -= 280
            current_boxes.append([x1, y1, x2, y2, class_id])

    # --- model2 좌표 처리 (640 -> 1280 스케일 + crop 보정) ---
    for result in detected2:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            class_id = int(box.cls.item())
            x1 = int(x1 * 2)
            x2 = int(x2 * 2)
            y1 = int(y1 * 2) - 280
            y2 = int(y2 * 2) - 280
            current_boxes.append([x1, y1, x2, y2, class_id])

    cv2.namedWindow('Editor')
    cv2.setMouseCallback('Editor', mouse_callback)

    while True:
        display_frame = frame.copy()
        list1 = [i[-1] for i in current_boxes]
        draw_boxes(display_frame, list1)

        if drawing and start_pos and end_pos:
            cv2.rectangle(display_frame, start_pos, end_pos, (255, 255, 0), 2)
            cv2.putText(display_frame, "Drawing...", (start_pos[0], start_pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)


        cv2.imshow('Editor', display_frame)
        cv2.imshow("frame", color_frame)
        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC
            break
        elif key == ord('a'):
            print(sorted(list1))
            if list1.count(1) == 2 and list1.count(2) == 2 and list1.count(3) == 2 and list1.count(4) == 2 and list1.count(5) == 2 and list1.count(6) == 2 and list1.count(0) == 2 and list1.count(7) == 1 and list1.count(8) == 1 and list1.count(9) == 1 and list1.count(10) == 1:
                save_labels(image_file, current_boxes, output_folder_path, w, h, top_pad, bottom_pad)
                cv2.imwrite(os.path.join(output_folder_path, image_file), gr)
                cv2.imwrite(os.path.join(output_color_folder_path, image_file), color_frame)
                print(f"Saved {image_file} and labels.")
                break
        elif key == ord('b'):
            break
        elif key == ord('q'):
            en = True
            break
        elif key == ord('u'):
            if current_boxes:
                removed = current_boxes.pop()
                print(f"Undo: removed box {removed}")

cv2.destroyAllWindows()
