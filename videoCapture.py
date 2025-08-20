import cv2
import os

save_path = 'recorded_video4.avi'  # avi 확장자 권장
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # MJPG 코덱
fps = 15  # 초당 프레임 수
frame_size = (1280, 720)  # 저장할 영상 크기

cap = cv2.VideoCapture(2)  # 카메라 열기
if not cap.isOpened():
    print("카메라 열기 실패")
    exit()

# 해상도 설정 (카메라가 지원해야 작동)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

out = cv2.VideoWriter(save_path, fourcc, fps, frame_size)

print("실시간 영상 녹화 시작. ESC 누르면 종료.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("프레임 읽기 실패")
        break

    # 프레임 크기 조절은 꼭 필요할 때만
    frame_resized = cv2.resize(frame, frame_size)

    cv2.imshow('Recording', frame_resized)
    out.write(frame_resized)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC키
        print("녹화 종료")
        break

cap.release()
out.release()
cv2.destroyAllWindows()
