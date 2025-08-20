import os

# 생성된 이미지와 라벨 파일 경로
image_folder = 'Dataset13/val/images'
label_folder = 'Dataset13/val/labels'

# 이미지 파일 목록과 라벨 파일 목록
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
label_files = [f for f in os.listdir(label_folder) if f.lower().endswith('.txt')]

# 이미지 파일과 라벨 파일의 이름이 정확히 일치하는지 확인
for image_file in image_files:
    base_name = os.path.splitext(image_file)[0]  # 확장자 제외한 파일 이름
    label_file = base_name + '.txt'  # 라벨 파일의 이름은 이미지 파일 이름과 동일해야 함
    
    if label_file in label_files:
        pass
        #print(f"파일 매칭 성공: {image_file} <-> {label_file}")
    else:
        print(f"파일 매칭 실패: {image_file}에 해당하는 라벨 파일이 없습니다.")
image_files = [i[:-4] for i in image_files]
label_files = [i[:-4] for i in label_files]

print(set(image_files)-set(label_files))
print(set(label_files)-set(image_files))
print(len(set(image_files)))
print(len(set(label_files)))