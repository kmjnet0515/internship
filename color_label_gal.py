import cv2
import numpy as np
import os

folder_path = 'saved_frames50'
output_path = 'saved_frames51'  # txt 파일도 같은 폴더에 저장

image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
def drawMarker(src_points, RGB_img):
    points = []
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            print(f"Point {len(points)}: ({x}, {y})")

            # 클릭한 점에 동그라미 표시
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Click 4 Points", clone)
    clone = RGB_img.copy()
    cv2.imshow("Click 4 Points", RGB_img)
    cv2.setMouseCallback("Click 4 Points", mouse_callback)
    
    while True:
        cv2.imshow("Click 4 Points", clone)
        key = cv2.waitKey(1) & 0xFF
        if len(points) == 4:
            break
    cv2.destroyAllWindows()
    return points
img_path = os.path.join(folder_path, image_files[0])
frame = cv2.imread(img_path)
src_points = np.float32(drawMarker([[],[],[],[]], frame))
polygon = np.array([src_points], dtype=np.int32)
for img_file in image_files:
    img_path = os.path.join(folder_path, img_file)
    frame = cv2.imread(img_path)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if frame is None:
        print(f"이미지 로드 실패: {img_path}")
        continue

    h_img, w_img = frame.shape[:2]

    # BGR → HSV 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    v = cv2.add(v, 50)  # V 채널 밝게
    v = np.clip(v, 0, 255)
    hsv = cv2.merge((h, s, v))
    # 빨간색 범위 (두 범위 필요)
    lower_red1 = np.array([120, 10, 10])
    upper_red1 = np.array([150, 100, 255])
    '''lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])'''

    

    # 마스크 생성
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1)
    '''mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)'''


    # 잡음 제거
    kernel = np.ones((1,1), np.uint8)
    mask_clean = cv2.morphologyEx(mask_red, cv2.MORPH_OPEN, kernel)
    kernel = np.ones((50,50), np.uint8)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # 컨투어 찾기
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_lines = []

    if contours:
        
        largest_two = sorted(contours, key=cv2.contourArea, reverse=True)
        count = 0
        for i, cnt in enumerate(largest_two):
            x, y, w, h = cv2.boundingRect(cnt)
            if w < 10 or h < 10:
                break
            result = cv2.pointPolygonTest(polygon, (x+w/2, y+h/2), False)
            print(x,y,result)
            if result < 0:
                continue
            if count == 4:
                break
            count += 1
            w += 6
            h += 6
            x -= 3
            y -= 3
            center_x = x + w // 2
            center_y = y + h // 2
            
            # YOLO 정규화
            x_center_norm = center_x / w_img
            y_center_norm = center_y / h_img
            w_norm = w / w_img
            h_norm = h / h_img

            # YOLO 라벨 문자열 (class id = 1)
            yolo_line = f"5 {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
            yolo_lines.append(yolo_line)

            # 표시용
            color = (0, 255, 0) if i == 0 else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
            print(f"{img_file} - Object {i+1}: {yolo_line}")
        if count < 4:
            continue
        # 결과 표시
        cv2.imshow("Mask", mask_red)
        cv2.imshow("Result", frame)
        key = cv2.waitKey(0)
        if key == ord('b'):  # ESC 누르면 종료
            continue
        # YOLO txt 파일 저장
        txt_file_name = os.path.splitext(img_file)[0] + '.txt'
        txt_path = os.path.join(output_path, txt_file_name)
        with open(txt_path, 'w') as f:
            for line in yolo_lines:
                f.write(line + '\n')
        
        cv2.imwrite(os.path.join(output_path, img_file), gray)

    

    

cv2.destroyAllWindows()
