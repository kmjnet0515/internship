import os

# 라벨 텍스트 파일이 들어있는 폴더 경로
label_folder = 'Dataset6/val/labels'  # 필요한 경로로 수정

# 폴더 내 모든 txt 파일 순회
for filename in os.listdir(label_folder):
    if filename.endswith('.txt'):
        file_path = os.path.join(label_folder, filename)
        with open(file_path, 'r') as f:
            lines = f.readlines()

        new_lines = []
        for line in lines:
            parts = line.strip().split()
            if len(parts) > 0 and parts[0] == '0':
                parts[0] = '7'  # 클래스 0 → 7 변경
            new_lines.append(' '.join(parts) + '\n')

        # 수정된 내용 덮어쓰기
        with open(file_path, 'w') as f:
            f.writelines(new_lines)

print("모든 txt 파일에서 클래스 0을 7로 변경 완료.")
