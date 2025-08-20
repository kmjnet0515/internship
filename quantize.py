import torch
import cv2
import os
import numpy as np
from tqdm import tqdm
import pathlib
from pathlib import Path

# Windows 경로 문제 방지
pathlib.PosixPath = pathlib.WindowsPath

# ===================
# ⚡ Calibration Dataset 정의
# ===================
class CalibrationDataset(torch.utils.data.Dataset):
    def __init__(self, img_dir, img_size=1280):
        self.img_dir = img_dir
        self.img_paths = [os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(('.jpg', '.png'))]
        self.img_size = img_size

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        img0 = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img0, (self.img_size, self.img_size))
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)
        img = img.transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255.0
        return img

# ===================
# ⚡ 모델 로드
# ===================
model_path = 'C:/Users/smpi9/Downloads/yolo_project/yolov5/runs/train/yolo_custom92/weights/best.pt'
model = torch.hub.load('./yolov5', 'custom', path=model_path, source='local', force_reload=True)

full_model = model.model.float().eval()

# ===================
# ⚡ CPU로 이동 후 정적 양자화 준비
# ===================
full_model = full_model.to('cpu')
full_model.qconfig = torch.quantization.get_default_qconfig('fbgemm')
torch.quantization.prepare(full_model, inplace=True)

# ===================
# ⚡ 보정용 데이터 로더 생성 및 Calibration 수행
# ===================
calib_dataset = CalibrationDataset('calibration_images', img_size=1280)
calib_loader = torch.utils.data.DataLoader(calib_dataset, batch_size=1, shuffle=False)

with torch.no_grad():
    for imgs in tqdm(calib_loader, desc="Calibrating"):
        imgs = imgs.to('cpu')
        full_model(imgs)

# ===================
# ⚡ 양자화 변환 적용
# ===================
torch.quantization.convert(full_model, inplace=True)

# ===================
# ⚡ 양자화 모델 저장
# ===================
torch.save({'model': full_model}, 'best_quantized_static3.pt')

print("✅ 정적 양자화 완료 및 저장 완료!")
