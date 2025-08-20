import os

# 확인할 폴더 경로
folder_path = "saved_labelstest2"  # 예: "C:/Users/username/Documents/test_folder"

# 조건 만족 파일 목록
files_with_large_floats = []

# 폴더 내 파일 반복
for filename in os.listdir(folder_path):
    if filename.endswith(".txt"):
        file_path = os.path.join(folder_path, filename)
        try:
            with open(file_path, 'r') as f:
                content = f.read()
                
                # 공백 기준으로 분리
                tokens = content.split()
                
                for token in tokens:
                    try:
                        value = float(token)
                        # 1 초과 & 소수만
                        if value > 1 and not value.is_integer():
                            files_with_large_floats.append(filename)
                            break  # 하나라도 찾으면 중단
                    except ValueError:
                        # 숫자가 아니면 패스
                        continue
        except Exception as e:
            print(f"파일 읽기 오류: {filename}, 오류: {e}")

# 결과 출력
print("1이 넘는 소수값을 포함한 파일 목록:")
for file in files_with_large_floats:
    print(file)
