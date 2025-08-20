import cv2
import numpy as np

# 전역 변수 (마우스 클릭 좌표 저장용)
clicked_point = None

# 마우스 콜백 함수
def mouse_callback(event, x, y, flags, param):
    global clicked_point
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_point = (x, y)

# 이미지 읽기
img_path = 'saved_frames2/frame2_00035.png'
frame = cv2.imread(img_path)

if frame is None:
    print("이미지 불러오기 실패")
    exit()

cv2.namedWindow("Image")
cv2.setMouseCallback("Image", mouse_callback)

while True:
    # 이미지 표시
    cv2.imshow("Image", frame)

    key = cv2.waitKey(1) & 0xFF

    # 엔터 키를 누르면 클릭 대기
    if key == 13:  # Enter 키
        print("마우스로 한 점을 클릭하세요.")
        clicked_point = None  # 초기화

        # 클릭될 때까지 기다리기
        while clicked_point is None:
            cv2.imshow("Image", frame)
            cv2.waitKey(1)

        x, y = clicked_point
        # 클릭한 좌표의 BGR 값
        bgr_value = frame[y, x]
        # HSV 변환 후 해당 점 HSV 값 추출
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hsv_value = hsv_frame[y, x]

        print(f"좌표: ({x}, {y})")
        print(f"BGR 값: {bgr_value}")
        print(f"HSV 값: {hsv_value}")

    # ESC 키 누르면 종료
    if key == 27:
        break

cv2.destroyAllWindows()
