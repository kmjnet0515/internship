import torch
import cv2
import os
import numpy as np
import pathlib
from pathlib import Path

# Windows 경로 문제 방지
pathlib.PosixPath = pathlib.WindowsPath

# ===================
# ⚡ Calibration Dataset 정의 (사용 안함)
# ===================
# ✅ 동적 양자화는 calibration dataset이 필요 없습니다. 따라서 이 부분은 삭제하거나 주석 처리해도 됩니다.

# ===================
# ⚡ 모델 로드
# ===================
model_path = 'C:/Users/smpi9/Downloads/yolo_project/yolov5/runs/train/yolo_custom105/weights/best.pt'
model = torch.hub.load('./yolov5', 'custom', path=model_path, source='local', force_reload=True)

# 원본 모델 가져오기
full_model = model.model.float().eval()

# ===================
# ⚡ 동적 양자화 적용
# ===================
quantized_model = torch.quantization.quantize_dynamic(
    full_model,  # 모델
    {torch.nn.Linear, torch.nn.Conv2d},  # 양자화할 레이어 타입
    dtype=torch.qint8
)

# ===================
# ⚡ 양자화 모델 저장
# ===================
torch.save({'model': quantized_model}, 'best_quantized_dynamic.pt')

print("✅ 동적 양자화 완료 및 저장 완료!")
