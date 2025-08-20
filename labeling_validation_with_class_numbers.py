import os
from collections import Counter

label_folder = "saved_framestest2"
valid = []

for filename in os.listdir(label_folder):
    if not filename.endswith(".txt"):
        continue

    file_path = os.path.join(label_folder, filename)
    with open(file_path, "r") as f:
        lines = f.readlines()

    class_counts = Counter()
    valid_classes = True

    for line in lines:
        parts = line.strip().split()
        if len(parts) < 1:
            continue
        try:
            cls = int(parts[0])
            if 0 <= cls <= 6:
                class_counts[cls] += 1
            else:
                valid_classes = False
                break
        except ValueError:
            valid_classes = False
            break

    # 각 클래스가 정확히 2번씩 있는지 확인
    if valid_classes and all(class_counts[c] == 2 for c in range(7)):
        valid.append(filename)

# 결과 출력
print(f"조건을 만족하는 파일 수: {len(valid)}")
for f in valid:
    print(f"✅ {f}")
