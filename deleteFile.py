import os

folder_path = "Dataset13/val/labels"  # 삭제할 폴더 경로
suffix = ".txt"  # 파일 확장자

start = 1
end = 2

for file_name in os.listdir(folder_path):
    # 파일 확장자 확인
    if not file_name.endswith(suffix):
        continue

    # 확장자 제거 후 뒤 4자리 숫자 추출
    base_name = file_name[:-len(suffix)]
    if len(base_name) < 4:
        continue
    number_str = base_name[-4:]

    # 숫자인지 확인
    if not number_str.isdigit():
        continue

    number = int(number_str)
    if start <= number <= end:
        file_path = os.path.join(folder_path, file_name)
        os.remove(file_path)
