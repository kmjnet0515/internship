import os

def find_invalid_txt_files(folder_path):
    result_files = []
    
    # 폴더 내 파일 확인
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):  # .txt 파일만
            file_path = os.path.join(folder_path, filename)
            
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    lines = f.readlines()
                    
                line_count = len(lines)
                
                # 조건: 0줄이거나 2줄 이상
                if line_count == 0 or line_count >= 2:
                    result_files.append((filename, line_count))
            
            except Exception as e:
                print(f"파일 읽기 오류: {filename}, 이유: {e}")
    
    return result_files


if __name__ == "__main__":
    folder = "Dataset13/train/labels"  # 검사할 폴더 경로
    invalid_files = find_invalid_txt_files(folder)
    
    if invalid_files:
        print("조건에 맞는 파일들:")
        for fname, cnt in invalid_files:
            print(f"- {fname}: {cnt}줄")
    else:
        print("조건에 맞는 파일이 없습니다.")
