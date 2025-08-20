#엔진 퀄리티 높음 : 200ms, 낮음 : 75ms, 보통 : 
import os
import cv2
import torch
import numpy as np
import pathlib
import sys
from pathlib import Path
pathlib.PosixPath = pathlib.WindowsPath
sys.path.append("C:/Users/smpi9/Downloads/yolo_project/ultralytics")
# 모델 로드
from ultralytics import YOLO
# 모델 로드
model = YOLO('C:/Users/smpi9/Downloads/yolo_project/ultralytics/runs/detect/best3(99).pt')


# 디바이스 설정
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = model.to(torch.device('cuda'))

# 폴더 설정
image_folder_path = 'new_frame14'
output_folder_path = 'new_saved_frame141'
output_color_folder_path = 'new_saved_frame143'
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
        return None
    color = frame.copy()
    h, w = frame.shape[:2]
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    target_size = 1280

    if h < target_size:
        total_pad = target_size - h
        top_pad = total_pad // 2
        bottom_pad = total_pad - top_pad
    else:
        top_pad = bottom_pad = 0

    img_padded = cv2.copyMakeBorder(img, top_pad, bottom_pad, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    img_resized = cv2.resize(img_padded, (1280, 1280))
    tensor_img = torch.from_numpy(img_resized).float() / 255.0
    tensor_img = tensor_img.permute(2, 0, 1).unsqueeze(0).to(device)
    results = model(tensor_img)

    '''detected_objects = results[0][results[0][:, 4] > 0.6]
    det_boxes = detected_objects[:, :4]
    scores = detected_objects[:, 4]
    classes = detected_objects[:, 5]
    indices = cv2.dnn.NMSBoxes(det_boxes.cpu().numpy(), scores.cpu().numpy(), score_threshold=0.5, nms_threshold=0.4)

    nms_detected_objects = []
    if len(indices) > 0:
        for i in indices.flatten():
            nms_detected_objects.append(detected_objects[i])'''
    return results, img_resized, w, h, top_pad, bottom_pad, color, img2
    
    #return nms_detected_objects, img_resized, w, h, top_pad, bottom_pad, color


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
    print(list1)
    
    if list1.count(1) == 2 and list1.count(2) == 2 and list1.count(3) == 2 and list1.count(4) == 2 and list1.count(5) == 2 and list1.count(6) == 2 and list1.count(0) == 2 :
        boxColor = (255,255,0)
    else:
        boxColor = (0,255,0)
    color_list = [(0,0,0) if list1.count(i) != 2 else boxColor for i in range(len(class_names))]
    for i, (x1, y1, x2, y2, cls_id) in enumerate(current_boxes):
        boxColor = color_list[cls_id]
        color = boxColor if i != selected_box_index else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{class_names[cls_id]}"
        cv2.putText(frame, label, (x1, max(y1 - 10, 0)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)


def mouse_callback(event, x, y, flags, param):
    global current_boxes, selected_box_index, drawing, start_pos, end_pos

    if event == cv2.EVENT_LBUTTONDOWN:
        # 박스 그리기 시작
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
            x1, y1 = start_pos
            x2, y2 = end_pos
            x1, x2 = sorted([x1, x2])
            y1, y2 = sorted([y1, y2])
            current_boxes.append([x1, y1, x2, y2, 8])  # 기본 클래스 0
            selected_box_index =len(current_boxes)-1
        start_pos = end_pos = None
    elif event == cv2.EVENT_MBUTTONDOWN:
        # 중간 버튼 클릭 시 박스 선택
        for i, (x1, y1, x2, y2, _) in enumerate(current_boxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                selected_box_index = i
                break
        else:
            selected_box_index = None
    elif event == cv2.EVENT_RBUTTONDOWN:
        # 오른쪽 클릭 시 박스 삭제 (선택된 박스가 없어도 마우스 위치에 있는 박스 삭제)
        for i, (x1, y1, x2, y2, _) in enumerate(current_boxes):
            if x1 <= x <= x2 and y1 <= y <= y2:
                del current_boxes[i]
                # 선택 박스 초기화
                selected_box_index = None
                break

    elif event == cv2.EVENT_MOUSEWHEEL:
        # 휠 돌리면 선택된 박스 클래스 변경
        if selected_box_index is not None:
            direction = 1 if flags > 0 else -1
            current_boxes[selected_box_index][4] = (current_boxes[selected_box_index][4] + direction) % len(class_names)




cout = 0
en = False
for image_file in image_files:
    image_path = os.path.join(image_folder_path, image_file)
    if int(image_path[-9:-4]) <= -1:
           continue
    if en:
        if cout < 10:
            cout += 1
            continue
        else:
            cout = 0
            en = False
    key = cv2.waitKey(1) & 0xFF
    if key == ord('b'):
        continue
    detected_objects, frame, w, h, top_pad, bottom_pad, color_frame, gr= process_image(image_path)
    frame = frame[280:1000, :]

    current_boxes = []
    for result in detected_objects:
        boxes = result.boxes  # Boxes object

        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            x1, y1, x2, y2 = map(int, (x1,y1,x2,y2))
            confidence = box.conf.item()
            class_id = int(box.cls.item())
            print(f"Box: ({x1:.2f}, {y1:.2f}) → ({x2:.2f}, {y2:.2f}), conf: {confidence:.2f}, class: {class_id}")
            current_boxes.append([x1,y1-280,x2,y2-280,class_id])
            #cv2.rectangle(frame, (x1,y1-280), (x2,y2-280), (0,0,255), 2)
            #cv2.putText(frame, model.names[class_id], (x1, y1 - 280-30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,0,255), 1)
    '''for det in detected_objects:
        x_center, y_center, width, height, conf, *class_probs = det
        y_center -= 280
        cls = int(torch.argmax(torch.tensor(class_probs)).item())
        if cls == 0:
            continue
        elif cls == 7:
            cls = 0
        x1 = int(x_center - width / 2)
        y1 = int(y_center - height / 2)
        x2 = int(x_center + width / 2)
        y2 = int(y_center + height / 2)
        current_boxes.append([x1, y1, x2, y2, cls])'''

    cv2.namedWindow('Editor')
    cv2.setMouseCallback('Editor', mouse_callback)
    
    while True:
        display_frame = frame.copy()
        list1 = [i[-1] for i in current_boxes]
        draw_boxes(display_frame, list1)
        
        # 드래그 중 시각화
        if drawing and start_pos and end_pos:
            cv2.rectangle(display_frame, start_pos, end_pos, (255, 255, 0), 2)
            cv2.putText(display_frame, "Drawing...", (start_pos[0], start_pos[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        if list1.count(1) == 2 and list1.count(2) == 2 and list1.count(3) == 2 and list1.count(4) == 2 and list1.count(5) == 2 and list1.count(6) == 2 and list1.count(0) == 2 :
            print("ok")
        else:
            print("no")
        cv2.imshow('Editor', display_frame)
        cv2.imshow("frame", color_frame)
        key = cv2.waitKey(1) & 0xFF

        # while 루프 안에 key 처리 부분 수정 예시
        if key == 27:  # ESC
            break
        elif key == ord('a') :#and list1.count(1) == 2 and list1.count(2) == 2 and list1.count(3) == 2 and list1.count(4) == 2 and list1.count(5) == 2 and list1.count(6) == 2 and list1.count(0) == 2:  # Enter
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
