import cv2
import numpy as np

# ==============================
# 1️⃣ 이미지 불러오기
image = cv2.imread("DataSet3/train/images/frame_00229.png")

if image is None:
    print("이미지를 불러오지 못했습니다. 경로와 파일명을 확인하세요.")
    exit()

h, w = image.shape[:2]
clone = image.copy()

# ==============================
# 2️⃣ 클릭한 좌표 저장용 리스트
points = []

def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        points.append([x, y])
        print(f"Point {len(points)}: ({x}, {y})")

        # 클릭한 점에 동그라미 표시
        cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Click 4 Points", clone)

# ==============================
# 3️⃣ 클릭용 윈도우
cv2.imshow("Click 4 Points", image)
cv2.setMouseCallback("Click 4 Points", mouse_callback)

print("네 개의 꼭짓점을 클릭하세요 (좌상, 우상, 우하, 좌하 순서 권장)")

while True:
    cv2.imshow("Click 4 Points", clone)
    key = cv2.waitKey(1) & 0xFF
    if len(points) == 4:
        break

cv2.destroyAllWindows()

# ==============================
# 4️⃣ 클릭한 좌표로 변환
src_points = np.float32(points)

# 변환 후 원하는 크기 (예: 원래 이미지 크기 유지)
dst_points = np.float32([
    [0, 0],
    [w, 0],
    [w, h],
    [0, h]
])

# ==============================
# 5️⃣ 변환 행렬과 보정
matrix = cv2.getPerspectiveTransform(src_points, dst_points)
result = cv2.warpPerspective(image, matrix, (w, h))
print(result.shape[:2])
# ==============================
# 6️⃣ 결과 보기
cv2.imshow("Warped (Front View)", result)

cv2.waitKey(0)
cv2.destroyAllWindows()

# ==============================
# 7️⃣ 결과 저장
cv2.imwrite("output.jpg", result)
print("완료! output.jpg로 저장되었습니다.")
