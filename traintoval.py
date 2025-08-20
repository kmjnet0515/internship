import os
import random
import shutil

# 경로 설정
train_images_dir = 'Dataset13/train/images'
train_labels_dir = 'Dataset13/train/labels'
val_images_dir = 'Dataset13/val/images'
val_labels_dir = 'Dataset13/val/labels'
'''
train_images_dir = 'detect/images/train'
train_labels_dir = 'detect/labels/train'
val_images_dir = 'detect/test/val'
val_labels_dir = 'detect/test/val' '''
# val 디렉토리 없으면 생성
os.makedirs(val_images_dir, exist_ok=True)
os.makedirs(val_labels_dir, exist_ok=True)

# train 이미지 파일 리스트 가져오기
image_files = [f for f in os.listdir(train_images_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# 1000장 무작위 선택
selected_images = random.sample(image_files, 5000)

for image_file in selected_images:
    # 이미지 경로
    src_image_path = os.path.join(train_images_dir, image_file)
    dst_image_path = os.path.join(val_images_dir, image_file)

    # 라벨 파일 이름 (확장자 변경)
    label_file = os.path.splitext(image_file)[0] + '.txt'
    src_label_path = os.path.join(train_labels_dir, label_file)
    dst_label_path = os.path.join(val_labels_dir, label_file)

    # 이미지 및 라벨 이동
    shutil.move(src_image_path, dst_image_path)
    if os.path.exists(src_label_path):
        shutil.move(src_label_path, dst_label_path)
    else:
        print(f"⚠️ 라벨 없음: {label_file}")
