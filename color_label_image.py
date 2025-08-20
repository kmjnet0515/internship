import cv2
import numpy as np
import os

folder_path = 'saved_frames3'
output_path = 'saved_frames3'  # txt 파일도 같은 폴더에 저장

image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

for img_file in image_files:
    img_path = os.path.join(folder_path, img_file)
    frame = cv2.imread(img_path)

    if frame is None:
        print(f"이미지 로드 실패: {img_path}")
        continue

    h_img, w_img = frame.shape[:2]

    # BGR → HSV 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 빨간색 범위 (두 범위 필요)
    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    # 노란색 범위
    lower_yellow = np.array([10, 100, 100])
    upper_yellow = np.array([40, 255, 255])

    # 마스크 생성
    mask_red1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask_red2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask_red = cv2.bitwise_or(mask_red1, mask_red2)

    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # 마스크 합치기
    mask_combined = cv2.bitwise_or(mask_red, mask_yellow)

    # 잡음 제거
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # 컨투어 찾기
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    yolo_lines = []

    if contours:
        # 면적 기준으로 상위 2개 선택
        largest_two = sorted(contours, key=cv2.contourArea, reverse=True)[:2]

        for i, cnt in enumerate(largest_two):
            x, y, w, h = cv2.boundingRect(cnt)
            w += 10
            h += 10
            x -= 5
            y -= 5
            center_x = x + w // 2
            center_y = y + h // 2
            
            # YOLO 정규화
            x_center_norm = center_x / w_img
            y_center_norm = center_y / h_img
            w_norm = w / w_img
            h_norm = h / h_img

            # YOLO 라벨 문자열 (class id = 1)
            yolo_line = f"2 {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"
            yolo_lines.append(yolo_line)

            # 표시용
            color = (0, 255, 0) if i == 0 else (255, 0, 0)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.circle(frame, (center_x, center_y), 5, color, -1)
            print(f"{img_file} - Object {i+1}: {yolo_line}")

    # YOLO txt 파일 저장
    txt_file_name = os.path.splitext(img_file)[0] + '.txt'
    txt_path = os.path.join(output_path, txt_file_name)
    with open(txt_path, 'w') as f:
        for line in yolo_lines:
            f.write(line + '\n')

    # 결과 표시
    cv2.imshow("Mask", mask_clean)
    cv2.imshow("Result", frame)

    key = cv2.waitKey(0)
    if key == 27:  # ESC 누르면 종료
        break

cv2.destroyAllWindows()
