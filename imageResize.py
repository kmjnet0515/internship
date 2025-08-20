import cv2

# 원본 이미지 경로
input_image_path = "new_saved_frame113/new_frame11_00000.png"

# 저장할 경로
output_image_path = "output_image_resized.png"

# 이미지 읽기
img = cv2.imread(input_image_path)
if img is None:
    raise ValueError("이미지를 읽을 수 없습니다.")

# 1280x720 → 640x360으로 정확히 절반 리사이즈
resized_img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)

# 저장
cv2.imwrite(output_image_path, resized_img)
print(f"저장 완료: {output_image_path}")
