import cv2
import numpy as np

# 카메라 열기
cap = cv2.VideoCapture(2)  # 카메라 번호 (보통 0)

if not cap.isOpened():
    print("카메라 열기 실패")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        break

    # BGR → HSV 변환
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # 파란색 범위 정의 (필요하면 조정)
    lower_blue = np.array([150, 0, 100])
    upper_blue = np.array([175, 255, 255])

    # 마스크 생성
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 잡음 제거 (모폴로지 연산)
    kernel = np.ones((5, 5), np.uint8)
    mask_clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_CLOSE, kernel)

    # 컨투어 찾기
    contours, _ = cv2.findContours(mask_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        # 가장 큰 컨투어 선택
        largest = max(contours, key=cv2.contourArea)

        # 외접 사각형
        x, y, w, h = cv2.boundingRect(largest)

        # 중심 좌표 계산
        center_x = x + w // 2
        center_y = y + h // 2

        # 사각형 & 중심점 그리기
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        # 정보 출력
        print(f"Center: ({center_x}, {center_y}), Width: {w}, Height: {h}")

    # 결과 표시
    cv2.imshow("Mask", mask_clean)
    cv2.imshow("Result", frame)

    # ESC(27) 키 누르면 종료
    if cv2.waitKey(1) & 0xFF == 27:
        break

# 자원 해제
cap.release()
cv2.destroyAllWindows()
