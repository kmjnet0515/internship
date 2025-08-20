import os
import cv2

# 원본 이미지 폴더 경로
input_dir = 'DataSet3/train/images'
# 새로운 이미지 저장할 폴더 경로
output_dir = 'DataSet4'

# 출력 폴더가 없으면 생성
os.makedirs(output_dir, exist_ok=True)

# 폴더 안의 모든 파일 가져오기
image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

for file_name in image_files:
    input_path = os.path.join(input_dir, file_name)
    output_path = os.path.join(output_dir, file_name)

    # 이미지 읽기
    img = cv2.imread(input_path)
    if img is None:
        print(f"이미지 불러오기 실패: {input_path}")
        continue

    # 흑백 변환
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 저장
    cv2.imwrite(output_path, gray_img)
    print(f"저장 완료: {output_path}")

print("모든 이미지 변환 및 저장 완료!")
