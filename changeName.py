import os

# 변경할 폴더 경로
folder_path = "new_saved_frame153"  # 예: "C:/Users/username/Documents/test_folder"

# 폴더 내 모든 파일 반복
for filename in os.listdir(folder_path):
    # 파일 확장자 추출
    name, ext = os.path.splitext(filename)
    
    # png 또는 txt 파일만 처리
    if ext.lower() in ['.png', '.txt']:
        # 새 파일명: 기존 이름 + 'kkk' + 확장자
        new_name = name + "color" + ext
        
        # 원래 전체 경로와 새 전체 경로
        src = os.path.join(folder_path, filename)
        dst = os.path.join(folder_path, new_name)
        
        # 이름 변경
        os.rename(src, dst)
        print(f"Renamed: {filename} -> {new_name}")

print("모든 파일 이름 변경 완료!")
