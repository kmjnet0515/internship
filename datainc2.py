import os
import cv2
import numpy as np

# 원본 이미지와 라벨 경로
image_folder = 'new_saved_frame111'
label_folder = 'new_saved_label111'
output_image_folder = 'Dataset13/train/images' 
output_label_folder = 'Dataset13/train/labels' 

os.makedirs(output_image_folder, exist_ok=True)
os.makedirs(output_label_folder, exist_ok=True)

image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def adjust_brightness(image, factor):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = np.clip(v.astype(np.float32) * factor, 0, 255).astype(np.uint8)
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def adjust_contrast(image, alpha):
    return cv2.convertScaleAbs(image, alpha=alpha, beta=0)

def process_image(image_path, label_path):
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    with open(label_path, 'r') as f:
        label_lines = f.readlines()

    base_name = os.path.splitext(os.path.basename(image_path))[0]
    count = 1

    # 1. 밝기 증가 (1.05 ~ 1.25)
    for i in range(5):
        factor = 1 + i * 0.05
        bright_image = adjust_brightness(image, factor)
        save_augmented_image(bright_image, label_lines, base_name, count)
        count += 1

    # 2. 밝기 감소 (0.95 ~ 0.75)
    for i in range(5):
        factor = 1 - (i + 1) * 0.05
        dark_image = adjust_brightness(image, factor)
        save_augmented_image(dark_image, label_lines, base_name, count)
        count += 1

    # 3. 대비 증가 (alpha: 1.1 ~ 1.5)
    for i in range(5):
        alpha = 1.1 + i * 0.1
        high_contrast = adjust_contrast(image, alpha)
        save_augmented_image(high_contrast, label_lines, base_name, count)
        count += 1

    # 4. 대비 감소 (alpha: 0.9 ~ 0.5)
    for i in range(5):
        alpha = 0.9 - i * 0.1
        low_contrast = adjust_contrast(image, alpha)
        save_augmented_image(low_contrast, label_lines, base_name, count)
        count += 1

def save_augmented_image(image, label_lines, base_name, count):
    img_name = f"{base_name}_{str(count).zfill(4)}.png"
    label_name = f"{base_name}_{str(count).zfill(4)}.txt"
    cv2.imwrite(os.path.join(output_image_folder, img_name), image, [cv2.IMWRITE_PNG_COMPRESSION, 0])
    with open(os.path.join(output_label_folder, label_name), 'w') as f:
        f.writelines(label_lines)


# 실행
for image_file in image_files:
    image_path = os.path.join(image_folder, image_file)
    label_file = os.path.splitext(image_file)[0] + ".txt"
    label_path = os.path.join(label_folder, label_file)

    if os.path.exists(label_path):
        process_image(image_path, label_path)

print("✅ 컬러 이미지 밝기 + 대비 증강 완료! (1장당 20장)")
