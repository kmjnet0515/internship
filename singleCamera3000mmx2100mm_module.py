print("⚡ 헤더 파일 실행 시작됨")

print("kornia import")
import kornia
print("▶ importing sys")
import sys
sys.path.append("C:/Users/smpi9/Downloads/yolo_project/ultralytics")
print("▶ importing os")
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

from ultralytics import YOLO
YOLOV5_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "YOLO5", "yolov5_local"))
sys.path.append(YOLOV5_PATH)
YOLOV5_PATH = os.path.normpath(os.path.join(BASE_DIR, "..", "yolov5"))
sys.path.append(YOLOV5_PATH)
#sys.path.append("C:/Users/Alijas/.cache/torch/hub/ultralytics_yolov5_master")
print("▶ importing np")
import numpy as np
print("▶ importing cv2")
import cv2
import orjson
print("▶ importing rs")
import pyrealsense2 as rs
print("▶ importing aruco")
from cv2 import aruco 
print("▶ importing json")
import json
print("▶ importing time")
import time
print("▶ importing ThreadPoolExecutor")
from concurrent.futures import ThreadPoolExecutor
print("▶ importing keyboard")
import keyboard
print("▶ importing flask")
from flask import Flask
print("▶ importing socket")
from flask_socketio import SocketIO
print("▶ importing asyncio")
import asyncio
print("▶ importing websocket")
import websockets
print("▶ importing torch")
import torch
print("▶ importing pathlib")
import pathlib
print("▶ importing math")
import math
print("▶ importing socket")
import socket
print("▶ importing struct")
import struct
print("▶ importing mp")
import mediapipe as mp
print("▶ importing device")

print("CUDA 사용 가능:", torch.cuda.is_available())
print("GPU 이름:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "없음")


print("⚡ 헤더내의  모든 모듈 로드 끝")

left_previous_static_hand_events = []
right_previous_static_hand_events = []


from fastapi import FastAPI, Response
from fastapi.responses import StreamingResponse
import threading
wall_width=300#테스트할 벽의 가로 
wall_height=210#테스트할 벽의 세로

138, 154

SETTING_FILE = "camera_setting_value.txt"

def load_settings():
    with open(SETTING_FILE, "r") as f:
        lines = f.read().strip().split("\n")

    # 1번째 줄: exposure, gain (int)
    left_exp, left_gain, right_exp, right_gain = map(int, lines[0].split())

    # 2번째 줄: left_src_points
    left_points_str = lines[1].split()
    left_src_points = np.float32([list(map(float, p.split(','))) for p in left_points_str])

    # 3번째 줄: right_src_points
    right_points_str = lines[2].split()
    right_src_points = np.float32([list(map(float, p.split(','))) for p in right_points_str])

    return left_exp, left_gain, right_exp, right_gain, left_src_points, right_src_points

# 사용 예시
left_exp, left_gain, right_exp, right_gain, left_src_points, right_src_points = load_settings()

print("left_exp:", left_exp)
print("left_gain:", left_gain)
print("right_exp:", right_exp)
print("right_gain:", right_gain)

print("left_src_points:\n", left_src_points)
print("right_src_points:\n", right_src_points)
projector_width=3840
projector_height=2160

half_wall_width=wall_width/2


left_dst_points = np.float32([#벽의 가로 세로 길이 
    [0, 0],
    [wall_width, 0],
    [wall_width, wall_height],
    [0, wall_height]
])
right_dst_points = np.float32([#벽의 가로 세로 길이 
    [138, 0],
    [wall_width, 0],
    [wall_width, wall_height],
    [138, wall_height]
])



prev_detected_list = []
frame_count = 0
YOLO_DETECT_INTERVAL = 5  # YOLO 추론 간격 (프레임 수 기준)

#------------------------------------------------------------------------------------#OPEN CV 분석 

Per_Frame=5 #1초에 몇프레임정도 사진을 찍나 

min_confidence=0.7 #최소한의 정확도 
min_confidence_person = 0.6
#-------------------------------------------------------------------------------------RGB-D 카메라로부터 프레임 받는 옵션
# 왼쪽 카메라용 pipeline-리얼 센스 d455
def init_camera():
    left_pipeline = rs.pipeline()
    left_config = rs.config()
    left_config.enable_device('339522300522')  # 중요!
    left_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, Per_Frame)
    left_config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, Per_Frame)
    left_align = rs.align(rs.stream.color)
    left_profile = left_pipeline.start(left_config)
    # 오른쪽 카메라용 pipeline
    right_pipeline = rs.pipeline()
    right_config = rs.config()
    right_config.enable_device('246322300435')  # 중요!
    right_config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, Per_Frame)
    right_config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, Per_Frame)
    right_profile = right_pipeline.start(right_config)
    right_align = rs.align(rs.stream.color)


    left_rgb_sensor = left_profile.get_device().query_sensors()[1]  # [1] = RGB sensor
    left_rgb_sensor.set_option(rs.option.enable_auto_exposure, False)
    left_rgb_sensor.set_option(rs.option.exposure, left_exp)  # 최대 밝게 (가능하면 800~1600 사이 실험)
    left_rgb_sensor.set_option(rs.option.gain, left_gain)       # 밝기 강화, 노이즈가 적당하면 200까지도 OK

    right_rgb_sensor = right_profile.get_device().query_sensors()[1]  # [1] = RGB sensor
    right_rgb_sensor.set_option(rs.option.enable_auto_exposure, False)
    right_rgb_sensor.set_option(rs.option.exposure, right_exp)  # 최대 밝게 (가능하면 800~1600 사이 실험)
    right_rgb_sensor.set_option(rs.option.gain, right_gain)       # 밝기 강화, 노이즈가 적당하면 200까지도 OK

    return left_pipeline, left_align, right_pipeline, right_align
#-----------------------------------------------------------------------------------------객체가 터치되었는지 판단하는 옵션

TCP_IP = "127.0.0.1"   
TCP_PORT = 2368      # LiDAR에서 송신하는 포트
BUFFER_SIZE = 1206


sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#sock.bind((UDP_IP, UDP_PORT))


Touched_Positions_Frame=[]#프레임별로 터치가 된 상대적 좌표 모음

tcp_socket=None

f = None
t = None
personId = 0
async def gen_frames():
    print("✅ Unreal 클라이언트가 연결되었습니다.")
    global left_src_points, right_src_points, left_dst_points, right_dst_points, left_previous_static_hand_events, right_previous_static_hand_events, f, t, personId
    left_pipeline, left_align, right_pipeline, right_align = init_camera()
    try:
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=2)
        Left_Color, Left_Depth, Left_Raw_Depth_Frame= await loop.run_in_executor(executor, Get_Frame, left_pipeline, left_align)
        Right_Color, Right_Depth, Right_Raw_Depth_Frame = await loop.run_in_executor(executor, Get_Frame, right_pipeline, right_align)

        
        Left_Zwall = Get_Zwall(Left_Depth, left_src_points, left_pipeline, left_align)#왼쪽 Z Wall 
        Right_Zwall = Get_Zwall(Right_Depth, right_src_points, right_pipeline, right_align)#오른쪽 Z Wall
        start = time.time()
        count = 0
        prev = []
        while True:
            end = time.time()
            count += 1
            if end-start > 1:
                start = time.time()
                print(f"현재프레임 : {count}")
                count = 0
            personId = 0
            future_left = loop.run_in_executor(executor, all_process_for_detect, left_pipeline, left_align, left_src_points, left_dst_points, Left_Zwall, left_previous_static_hand_events, "left")
            future_right = loop.run_in_executor(executor, all_process_for_detect, right_pipeline, right_align, right_src_points, right_dst_points, Right_Zwall, right_previous_static_hand_events, "right")

            # 두 작업이 모두 끝날 때까지 기다림 (병렬 실행 + 동시 대기)
            (left_RealFinal_data, img1, deletedList1), (right_RealFinal_data, img2, deletedList2) = await asyncio.gather(future_left, future_right)
            dst = np.float32([[0,0],[1280,0],[1280,720],[0,720]])
            matrix1 = cv2.getPerspectiveTransform(left_src_points, dst)
            #matrix2 = cv2.getPerspectiveTransform(right_src_points, dst)
            # 투시 변환 적용
            warped1 = cv2.warpPerspective(img1, matrix1, (1280, 720))
            #warped2 = cv2.warpPerspective(img2, matrix2, (1280, 720))
            #warped2=warped2[:,120:1280]
            #horizontal = np.hstack((warped1, warped2))
            batch_data = [] # left_batch_data와 right_batch_data 처리 후 실행
            #Realfinal_unique, armUnique =merge_and_deduplicate(horizontal,left_RealFinal_data, right_RealFinal_data, deletedList1, deletedList2)
            Realfinal_unique, armUnique, deletedListMerged =merge_and_deduplicate(warped1,left_RealFinal_data, right_RealFinal_data, deletedList1, deletedList2, left_previous_static_hand_events, right_previous_static_hand_events)

            t = (Realfinal_unique, armUnique, left_previous_static_hand_events, deletedListMerged)
            if f is not None:
                ret, buffer = cv2.imencode('.jpg', f)
                frame_bytes = buffer.tobytes()
                
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


            FinalArmData.count = 0
            

            '''threshold = 0.1
            unique = []
            for idx1, obj1 in enumerate(Realfinal_unique):
                is_duplicate = False
                for obj2 in prev:
                    x_diff = obj1.relative_x - obj2.relative_x
                    y_diff = obj1.relative_y - obj2.relative_y
                    # 👇 label까지 비교
                    if obj1.label == obj2.label and abs(x_diff) < threshold and abs(y_diff) < threshold:
                        is_duplicate = True
                        break
                if not is_duplicate:
                    unique.append(obj1)
            prev = Realfinal_unique'''
    finally:
        executor.shutdown(wait=True)
        left_pipeline.stop()
        right_pipeline.stop()
        print("🛑 카메라 종료 및 리소스 해제 완료")
        cv2.destroyAllWindows()
    
from fastapi.middleware.cors import CORSMiddleware
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발용으로는 "*" 허용
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
tete = None
@app.get("/video_feed")
def video_feed():
    return StreamingResponse(gen_frames(),
                             media_type='multipart/x-mixed-replace; boundary=frame')
@app.get("/get_text")
def get_text():
    global t, min_confidence, min_confidence_person
    r = r2 = r3 = r4= ""
    if t is not None:
        obj, arm, prev, delList = t
        for idx1, text in enumerate(obj):
            if text.label not in ['person', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']:
                r += f"id : {idx1}, timestamp : {time.strftime('%H:%M:%S')}\n" + str(text) + "\n\n"
        for text in arm:
            r2 += str(text)+"\n"
        for text in prev:
            r3 += str(text)+"\n"
        for idx1, text in enumerate(delList):
            if text.label not in ['person', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']:
                r4 += f"id : {idx1}, timestamp : {time.strftime('%H:%M:%S')}\n" + str(text) + "\n\n"
        
        return {"text": r, "text2" : r2, "text3" : r3, "text4" : r4, "text5" : f"conf : {min_confidence:.2f}, {min_confidence_person:.2f}"}




    




epsilon = 3  # 오차 허용값: 약 3cm 정도
block_thickness = 1.5  # 블록 두께: 2cm
brightness = 0.2


class customObject:
    def __init__(self, x1, y1, x2, y2, conf):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
        self.conf = conf


class TouchObject(customObject):
    def __init__(self, center_x, center_y, label, isTouch, personId,x1, y1, x2, y2, conf):
        super().__init__(x1,y1,x2,y2, conf)
        self.center_x = center_x
        self.center_y = center_y
        self.label = label
        self.isTouch = isTouch
        self.personId = personId

class DetectedObject(customObject):
    def __init__(self, center_x, center_y, label, personId, x1, y1, x2, y2, conf):
        super().__init__(x1,y1,x2,y2, conf)
        self.center_x = center_x
        self.center_y = center_y
        self.label = label
        self.personId = personId



class FinalFowardObject(customObject):
    def __init__(self, relative_x,relative_y,center_x,center_y ,label, isTouch, personId , x1, y1, x2, y2, conf):
        super().__init__(x1,y1,x2,y2, conf)
        self.relative_x = relative_x
        self.relative_y = relative_y
        self.center_x=center_x
        self.center_y=center_y
        self.label = label
        self.isTouch = isTouch
        self.personId = personId
    def __str__(self):
        return f"상대좌표 : ({self.relative_x:.2f},{self.relative_y:.2f}), 중간좌표 ({self.center_x:.2f}, {self.center_y:.2f})\n식별자 : {self.label}, 사람ID : {self.personId}"

class RealFinalObject(customObject):#프레임 필터후의  프레임에서의  데이터 구조체
    def __init__(self,IsAttached,relative_x,relative_y, center_x,center_y,label, isTouch,personId, x1, y1, x2, y2, conf):
        super().__init__(x1,y1,x2,y2, conf)
        self.IsAttached=IsAttached
        self.relative_x = relative_x
        self.relative_y = relative_y
        self.center_x=center_x
        self.center_y=center_y
        self.label = label
        self.isTouch = isTouch
        self.personId = personId
    def __str__(self):
        return f"부착여부 : {self.isTouch}, 상대좌표 : ({self.relative_x:.2f},{self.relative_y:.2f})\n중간좌표 ({self.center_x:.2f}, {self.center_y:.2f}),식별자 : {self.label}, 사람ID : {self.personId}"
class FinalArmData:
    count = 0
    def __init__(self, x1,y1,x2,y2):
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1
        FinalArmData.count += 1
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    def setter(self):
        FinalArmData.count = 0
    def getter(self):
        return FinalArmData.count
    def __str__(self):
        return f"(x1 : {self.x1:.2f}, y1 : {self.y1:.2f}, x2 : {self.x2:.2f}, y2 : {self.y2:.2f})"


class Person:
    def __init__(self, x1,y1,x2,y2):
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2
    def __str__(self):
        return f"사람 좌표. 좌상 : ({self.x1},{self.y1}) 우상 : ({self.x2},{self.y2})"



MARKER_ID_MAP = {
    "left_top": 0,
    "right_top": 1,
    "right_bottom": 2,
    "left_bottom": 3
}

ZWall=None



def Get_Frame(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    depth_frame = aligned_frames.get_depth_frame()
    color_frame = aligned_frames.get_color_frame()

    print("🌈 color_frame:", color_frame.get_frame_number(), "🌊 depth_frame:", depth_frame.get_frame_number())

    if not depth_frame or not color_frame:
        print("⚠️ 프레임이 비어 있음 (color or depth)")

    # numpy 배열로 변환 (복사본)
    color = np.asanyarray(color_frame.get_data())
    depth = np.asanyarray(depth_frame.get_data())
    print("frame data out")
    return color, depth, depth_frame


def Detecte_Marker(RGB_img, src_points, list1):
    cor1, cor2, cor3, cor4 = list1

    gray = cv2.cvtColor(RGB_img, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)

    '''
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
    parameters = aruco.DetectorParameters_create()
    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 1
    parameters.adaptiveThreshConstant = 5
    parameters.minMarkerPerimeterRate = 0.015
    parameters.polygonalApproxAccuracyRate = 0.02
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    parameters.maxErroneousBitsInBorderRate = 0.5
    parameters.errorCorrectionRate = 0.6
    '''
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)

    # DetectorParameters_create() → DetectorParameters() 로 변경
    parameters = cv2.aruco.DetectorParameters()

    parameters.adaptiveThreshWinSizeMin = 3
    parameters.adaptiveThreshWinSizeMax = 23
    parameters.adaptiveThreshWinSizeStep = 1
    parameters.adaptiveThreshConstant = 5
    parameters.minMarkerPerimeterRate = 0.015
    parameters.polygonalApproxAccuracyRate = 0.02

    # cornerRefinementMethod 설정 시 enum 값 사용 (변경 없음)
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX

    parameters.maxErroneousBitsInBorderRate = 0.5
    parameters.errorCorrectionRate = 0.6
    print("✅ OpenCV version:", cv2.__version__)

    print("🧪 detectMarkers 호출 직전")
    try:
        corners, ids, _ = aruco.detectMarkers(equalized, aruco_dict, parameters=parameters)
        print("✅ detectMarkers 호출 완료")  # ← 이게 출력되어야 정상
    except Exception as e:
        print("❌ detectMarkers 에러:", e)
  


    output_img = RGB_img.copy()

    if ids is not None and len(ids) == 4:
        print("✅ 인식된 마커 개수:", len(ids))
        for i, marker_id in enumerate(ids.flatten()):
            print(f"  - ID {marker_id}의 코너 좌표:")
            for j, pt in enumerate(corners[i][0]):
                print(f"    > 코너 {j}: (x={pt[0]:.1f}, y={pt[1]:.1f})")
            
            # 각 마커의 사각형 좌표
            marker_corners = corners[i][0].astype(int)

            # 각 마커 개별 사각형 그리기
            cv2.polylines(output_img, [marker_corners.reshape(-1, 1, 2)], isClosed=True, color=(0, 255, 0), thickness=3)

            # ID 텍스트 표시
            center_x = int(marker_corners[:, 0].mean())
            center_y = int(marker_corners[:, 1].mean())
            cv2.putText(output_img, f"ID:{marker_id}", (center_x - 20, center_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.imshow(f"Markers (각각 표시){ids}", output_img)
        cv2.waitKey(1)
    else:
        print("❌ 마커를 찾지 못했습니다.")
        src_points = np.float32(drawMarker(src_points, RGB_img))
        return src_points

    # ID → corners 매핑
    id_to_corners = {int(marker_id): corners[i][0] for i, marker_id in enumerate(ids.flatten())}

    #try:
    src_points[0] = id_to_corners[cor1][0]
    src_points[1] = id_to_corners[cor2][1]
    src_points[2] = id_to_corners[cor3][2]
    src_points[3] = id_to_corners[cor4][3]
    '''except KeyError as e:
        print(f"⚠️ 마커 ID {e}가 인식되지 않았습니다.")
        return'''

    print("LeftTop: {} , RightTop: {} , RightBottom: {} , LeftBottom: {} ".format(
        src_points[0], src_points[1], src_points[2], src_points[3]))
    

    return src_points

def drawMarker(src_points, RGB_img):
    points = []
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append([x, y])
            print(f"Point {len(points)}: ({x}, {y})")

            # 클릭한 점에 동그라미 표시
            cv2.circle(clone, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow("Click 4 Points", clone)
    clone = RGB_img.copy()
    cv2.imshow("Click 4 Points", RGB_img)
    cv2.setMouseCallback("Click 4 Points", mouse_callback)
    
    while True:
        cv2.imshow("Click 4 Points", clone)
        key = cv2.waitKey(1) & 0xFF
        if len(points) == 4:
            break
    cv2.destroyAllWindows()
    return points



#MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR,"..", "YOLO5", "yolov5_local", "runs", "train", "fwu_1280_quick","weights","best.pt"))
#MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR,"..","YOLO5","yolov5_local","runs","train","Minzae","best.pt"))


#MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR,"..","yolov5","runs","train","yolo_7nc3","weights","best.pt"))
MODEL_PATH = os.path.normpath(os.path.join(BASE_DIR,"ultralytics","runs","detect","yolo_7nc44","weights","best.pt"))

'''YAML_PATH=os.path.normpath(os.path.join(BASE_DIR,"..","YOLO5","yolov5_local", "custom.yaml"))
YAML5S_PATH=os.path.normpath(os.path.join(BASE_DIR,"..","YOLO5","yolov5_local","models","yolov5s.yaml"))'''
#YAML_PATH=os.path.normpath(os.path.join(BASE_DIR,"..", "YOLO5", "yolov5_local","runs", "train", "Minzae", "data.yaml"))



pathlib.PosixPath = pathlib.WindowsPath

if not os.path.exists(MODEL_PATH):
    print(f"❌ 모델 파일 없음: {MODEL_PATH}")
else:
    print(f"📦 모델 파일 존재 확인: {MODEL_PATH}")    

'''if os.path.exists(YAML_PATH):
    print(f"📦 yaml 파일 존재 확인: {YAML_PATH}") '''   

#device = select_device("cuda" if torch.cuda.is_available() else "cpu")
#checkpoint = torch.load(MODEL_PATH, map_location='cpu')
#print(type(checkpoint))

print("⚡ 모델 로딩  시작")
#model=torch.hub.load('ultralytics/yolov5','custom', path=MODEL_PATH, force_reload=True)


#model = DetectMultiBackend(MODEL_PATH,'cuda',data=YAML_PATH)
#correct_model=load_model(MODEL_PATH,YAML5S_PATH,5)
#cfg_path = YAML_PATH  # 학습 시 사용한 구조 yaml
#NUM_CLASSES = 5  # ← 반드시 학습 당시 클래스 수와 동일하게 설정

# 1. 모델 구조 정의
#model = Model(cfg_path, ch=3, nc=NUM_CLASSES)

# 2. 가중치 로드
#model.load_state_dict(torch.load(MODEL_PATH, map_location='cuda'))

# 3. GPU로 이동 + 추론 모드
#model.to('cuda').eval()
print("⚡ 모델 로딩  시작")

#model = YOLO('C:/Users/smpi9/Downloads/yolo_project/ultralytics/runs/detect/yolo_7nc413/weights/best.pt')
model = YOLO('C:/Users/smpi9/Downloads/yolo_project/ultralytics/runs/detect/best3(99).pt')

model2 = YOLO("yolo11m-pose.pt")  
#model2.to(torch.device('cuda'))

model2.classes = [0]
print(type(model))

#model = torch.load(MODEL_PATH, map_location='cuda')
if model:
    print("⚡ 모델   로딩  완료")

print('''model info''')
print(model.names)
print(len(model.names))
#model.eval()

#model_EXP = torch.load(MODEL_PATH, map_location='cuda')

#print(model_EXP['model'].names)  # 클래스 이름
#print(model_EXP['model'].nc)     # 클래스 수
#print(model_EXP['model'].model[-1])


color = (0, 0, 255)
radius = 10
thickness = -1 






'''def preprocess_for_yolo_gpu(image, src_points, target_size=1280, device='cuda'):
    # image: BGR numpy array (H,W,C)
    # src_points: polygon points for masking
    
    # 1. numpy → torch tensor (BGR -> RGB), CHW, float, [0,1]
    img_tensor = torch.from_numpy(image).to(device).permute(2,0,1).float() / 255.0  # (C,H,W)
    
    # 2. Create mask tensor on GPU
    # 먼저 빈 mask 생성
    mask = torch.zeros(img_tensor.shape[1:], dtype=torch.uint8, device=device)  # (H,W)
    # src_points numpy → tensor, int64
    poly = torch.tensor(src_points, dtype=torch.int32, device=device)
    # fill polygon: kornia에서는 fill_poly가 없음. OpenCV CUDA를 직접 쓰기 어렵다면 CPU에서 mask 만들고 gpu로 올려도 괜찮음.
    # 여기서는 CPU에서 mask 만들고 GPU로 옮기는 혼합방식 (최소 부담)
    
    # CPU에서 mask 생성
    mask_cpu = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_cpu, [np.int32(src_points)], 255)
    mask = torch.from_numpy(mask_cpu).to(device)
    
    # 3. 마스크 적용 (mask가 255이므로 1로 정규화)
    masked = img_tensor * (mask.float() / 255.0).unsqueeze(0)  # (C,H,W)
    
    # 4. 그레이스케일 변환 (RGB -> Grayscale)
    gray = kornia.color.rgb_to_grayscale(masked.unsqueeze(0))  # (1,1,H,W)
    current_brightness = gray.mean().item()
    target_brightness = brightness

    # scale factor 계산
    if current_brightness > 0:
        scale = target_brightness / current_brightness
    else:
        scale = 1.0  # 밝기가 0일 경우는 그대로

    # 밝기 보정
    #gray = torch.clamp(gray * scale, 0.0, 1.0)  # (1,1,H,W), 평균 밝기가 0.2 근처로 조정됨
    #changed_brightness = gray.mean().item()
    #print(f"현재 평균 밝기 : {current_brightness}, 변환된 평균 밝기 {changed_brightness}")
    # 5. 패딩 처리: height 기준으로 padding 필요시
    h, w = gray.shape[2], gray.shape[3]
    if h < target_size:
        pad_total = target_size - h
        # top_pad, bottom_pad 나누기 (원하면)
        top_pad = 0
        bottom_pad = pad_total
        gray = torch.nn.functional.pad(gray, (0, 0, top_pad, bottom_pad), mode='constant', value=0)
    
    # 6. 리사이즈
    gray_resized = kornia.geometry.resize(gray, (target_size, target_size))  # (1,1,H,W)
    
    # 7. 3채널로 변환 (복제)
    gray_3ch = gray_resized.repeat(1, 3, 1, 1)  # (1,3,H,W)
    img_for_show = gray_3ch[0].permute(1, 2, 0).cpu().numpy()  # H, W, C
    # 0~1 float → 0~255 uint8 변환
    img_for_show = (img_for_show * 255).astype(np.uint8)
    return gray_3ch, img_for_show'''



def preprocess_for_yolo_gpu(image, src_points, target_size=1280, device='cuda'):
    # image: BGR numpy array (H,W,C)
    # src_points: polygon points for masking

    # 1. BGR → RGB → torch tensor
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img_tensor = torch.from_numpy(image_rgb).to(device).permute(2, 0, 1).float() / 255.0  # (C,H,W)

    # 2. Create mask on CPU → transfer to GPU
    mask_cpu = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_cpu, [np.int32(src_points)], 255)
    mask = torch.from_numpy(mask_cpu).to(device)  # (H,W)

    # 3. Apply mask
    masked = img_tensor * (mask.float() / 255.0).unsqueeze(0)  # (C,H,W)

    # 4. Padding (height only)
    h, w = masked.shape[1], masked.shape[2]
    if h < target_size:
        pad_total = target_size - h
        masked = torch.nn.functional.pad(masked, (0, 0, 0, pad_total), mode='constant', value=0)

    # 5. Resize
    masked = masked.unsqueeze(0)  # (1,3,H,W)
    resized = kornia.geometry.resize(masked, (target_size, target_size))  # (1,3,H,W)


    return resized, image



lock = threading.Lock()

model2.eval()
eee=0
def DetectAndGetData(img, src_points, where):
    Detected_list = []
    global eee, personId
    eee+=1
    process_img_gpu,img_for_show = preprocess_for_yolo_gpu(img, src_points, device='cuda')
    with torch.no_grad():
        try:
            results = model(process_img_gpu)  # 모델도 GPU에 올라가 있어야 합니다.

            for det in results:
                boxes = det.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    x1, y1, x2, y2 = map(int, (x1,y1,x2,y2))
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    if confidence < min_confidence:
                        continue
                    print(x1,y1,x2,y2,class_id,confidence)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    label = model.names[int(class_id)]
                    Detected_list.append(DetectedObject(cx, cy, label, None, x1, y1, x2, y2, confidence))


            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
            gray_3ch = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)  # (H, W, 3)

            results2 = model2.predict(gray_3ch[...,::-1])
            for r in results2:
                keypoints = r.keypoints
                kpts = keypoints.xy.cpu().numpy()   # shape: (num_people, 17, 2)
                confs = keypoints.conf.cpu().numpy()  # shape: (num_people, 17)
                
                bboxes = r.boxes.xyxy.cpu().numpy().astype(int)
                box_confs = r.boxes.conf.cpu().numpy()
                
                for person_idx, person_kpts in enumerate(kpts):
                    if box_confs[person_idx] < min_confidence_person:  # 예: 0.6 이하이면 무시
                        continue
                    # 바운딩 박스 정보
                    x1, y1, x2, y2 = bboxes[person_idx]
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2
                    personId += 1
                    Detected_list.append(DetectedObject(cx, cy, "person", person_idx + (50 if where == 'left' else 0), x1, y1, x2, y2, box_confs[person_idx]))
                    # 키포인트 정보
                    person_confs = confs[person_idx]
                    hand_labels = ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']
                    
                    # 필요한 관절 인덱스 (팔꿈치: 7,8 / 손목: 9,10)
                    for idx in [7, 8, 9, 10]:
                        if idx < len(person_confs) and person_confs[idx] >= min_confidence_person:
                            x, y = map(int, person_kpts[idx])
                            if 0 <= x < 1280 and 0 <= y < 720:
                                Detected_list.append(DetectedObject(x, y, hand_labels[idx - 7], person_idx + (50 if where == 'left' else 0) ,x, y, x, y, person_confs[idx]))
                            
        except:
            results = None
            results2 = None

    
            
    '''for det in detections:
        x1, y1, x2, y2, conf, cls = det.tolist()
        

        if conf < min_confidence:
            continue

       

        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        label = model.names[int(cls)]
        #print(model.names)
        #print(f"cx: {cx} ,cy : {cy}, label : {label}")
        Detected_list.append(DetectedObject(cx, cy, label, None, x1, y1, x2, y2))'''

    return Detected_list,img_for_show






async def send_data(Final_DataSet):
    uri = "ws://127.0.0.1:8765"

    async with websockets.connect(uri) as websocket:
        batch_data = []

        for Data in Final_DataSet:
            data_dict = {
                
                "object_type": Data.label,
                "Center_x": Data.center_x,
                "Center_y": Data.center_y,
                "timestamp": time.time()
            }

            if hasattr(Data, "width"):
                data_dict["width"] = Data.width
                data_dict["height"] = Data.height
                data_dict["angle"] = Data.angle

            batch_data.append(data_dict)

        await websocket.send(json.dumps(batch_data))



def filter_and_print_within_bounds(detected_objects, src_pts, dst_pts):
    Relative_List=[]
    matrix = cv2.getPerspectiveTransform(src_pts, dst_pts)

    for obj in detected_objects:
        points = [
            [obj.center_x, obj.center_y],
            [obj.x1, obj.y1],
            [obj.x2, obj.y2]
        ]
        pts_array = np.array(points, dtype='float32').reshape(-1, 1, 2)
        transformed_pts = cv2.perspectiveTransform(pts_array, matrix)  # shape: (3, 1, 2)
        (tx, ty) = transformed_pts[0][0]
        (x1, y1) = transformed_pts[1][0]
        (x2, y2) = transformed_pts[2][0]

        # 상대 좌표로 변환
        relative_tx, relative_ty = tx / wall_width, ty / wall_height
        relative_x1, relative_y1 = x1 / wall_width, y1 / wall_height
        relative_x2, relative_y2 = x2 / wall_width, y2 / wall_height
        '''pt = np.array([[[obj.center_x, obj.center_y]]], dtype='float32')
        transformed = cv2.perspectiveTransform(pt, matrix)[0][0]
        tx, ty = transformed
        relative_x = tx / wall_width
        relative_y = ty / wall_height
        pt2 = np.array([[[obj.x1, obj.y1]]], dtype='float32')
        transformed2 = cv2.perspectiveTransform(pt2, matrix)[0][0]
        x1, y1 = transformed2
        x1 = x1 / wall_width
        y1 = y1 / wall_height
        pt3 = np.array([[[obj.x1, obj.y1]]], dtype='float32')
        transformed3 = cv2.perspectiveTransform(pt3, matrix)[0][0]
        x2, y2 = transformed3
        x2 = x2 / wall_width
        y2 = y2 / wall_height'''
        #print(f"📌 범위 내 탐지: Label={obj.label}, Center=({relative_tx:.2f}, {relative_ty:.2f})")
        Relative_List.append(FinalFowardObject(relative_tx,relative_ty,obj.center_x,obj.center_y,obj.label, obj.isTouch, obj.personId,relative_x1,relative_y1, relative_x2, relative_y2, obj.conf))

    return Relative_List

def Get_Zwall(aligned_depth_frame, src_points, pipeline, align):
    _, initial_depth_image, initial_depth_frame = Get_Frame(pipeline, align)

    # 전체 영역을 컬러맵으로 시각화
    current_depth_colormap = cv2.applyColorMap(
        cv2.convertScaleAbs(initial_depth_image, alpha=0.03),
        cv2.COLORMAP_JET
    )
    cv2.imshow('zwall_full', current_depth_colormap)

    cv2.waitKey(1000)
    cv2.destroyAllWindows()

    return initial_depth_image



    zwall_values = []
    for y in range(720):
        for x in range(1280):
            if mask_bool[y, x]:
                z_m = aligned_depth_frame.get_distance(x, y)
                if z_m > 0:
                    zwall_values.append(z_m * 100)  # ✅ cm로 변환

    if len(zwall_values) == 0:
        print("❌ ZWall 계산 실패: 유효한 depth 값 없음")
        return 0.0

    median_z_cm = np.median(zwall_values)
    print(f"✅ ZWall 계산 완료: {median_z_cm:.1f} cm")
    return median_z_cm

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2,
                    enable_segmentation=False, min_detection_confidence=0.9)
mp_drawing = mp.solutions.drawing_utils

def GetIsTouch(frame, depth_image, Detected_List, ZWall):
    global tete
    tete=""
    Filtered_List = []
    for obj in Detected_List:
        
        center_x, center_y, label,x1,y1,x2,y2 = obj.center_x, obj.center_y, obj.label, obj.x1, obj.y1, obj.x2, obj.y2
        current_depth_value = depth_image[center_y, center_x]
        current_distance = current_depth_value * 0.001
        initial_depth_value = ZWall[center_y, center_x]
        initial_distance = initial_depth_value * 0.001
        depth_diff = initial_distance - current_distance
        if initial_distance < 0.001 or current_distance < 0.001:
            continue
        
        max_allowed = 0.2
        '''if label in ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']:
            max_allowed = 0.03'''
        tete += (f"label : {label}, 초기 : {initial_distance:.2f}, 현재 : {current_distance:.2f}, 차이 : {depth_diff:.2f}")
        Filtered_List.append(TouchObject(center_x, center_y, label, abs(depth_diff) < max_allowed, obj.personId, x1, y1, x2, y2, obj.conf))
    return Filtered_List
    
def make_key(obj):
    return (round(obj.relative_x * 50), round(obj.relative_y * 50), obj.label)
    # 0.02 tolerance를 위해 1/0.02 = 50 배율 사용
def is_overlapping(box1, box2):
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 겹치지 않는 조건
    if x1_max <= x2_min or x2_max <= x1_min:
        return False
    if y1_max <= y2_min or y2_max <= y1_min:
        return False

    # 위 조건에 모두 해당하지 않으면 겹침
    return True
def filter_new_before_StaticAndHand_events(current_events, Color, previous_static_hand_events):
    after_frame_datas = []
    next_previous_static_hand_events = []
    deletedList = []
   
    if len(previous_static_hand_events) > 0:
        # 1. 현재 프레임의 객체 처리
        for obj1 in current_events:
            if obj1.label in ['person', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']:
                after_frame_datas.append(
                    RealFinalObject(True, obj1.relative_x, obj1.relative_y,
                                    obj1.center_x, obj1.center_y, obj1.label, obj1.isTouch,
                                    obj1.personId, obj1.x1, obj1.y1, obj1.x2, obj1.y2, obj1.conf)
                )
                continue

            was_occluded = False
            for obj2 in previous_static_hand_events:
                if is_overlapping((obj1.x1, obj1.y1, obj1.x2, obj1.y2),
                                (obj2.x1, obj2.y1, obj2.x2, obj2.y2)):
                    was_occluded = True
                    next_previous_static_hand_events.append(obj1)
                    break
            if not was_occluded:
                newObj = RealFinalObject(True, obj1.relative_x, obj1.relative_y,
                                        obj1.center_x, obj1.center_y, obj1.label,
                                        obj1.isTouch, obj1.personId,
                                        obj1.x1, obj1.y1, obj1.x2, obj1.y2, obj1.conf)
                after_frame_datas.append(newObj)
                next_previous_static_hand_events.append(newObj)

        # 2. 이전 프레임에 있던 객체 중 가려진 것 유지 or 제거 판단
        for obj2 in previous_static_hand_events:
            is_in_current = any(
                is_overlapping((obj2.x1, obj2.y1, obj2.x2, obj2.y2),
                            (obj1.x1, obj1.y1, obj1.x2, obj1.y2)) and obj1.label == obj2.label
                for obj1 in current_events
            )
            #print(f"is_in_current:{is_in_current}")
            if is_in_current:
                continue  # 이미 현재 프레임에서 처리됨

            # 여전히 사람에 의해 가려졌는지 확인
            still_hidden = False
            for person_obj in current_events:
                if person_obj.label in ['person', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']:
                    if is_overlapping((obj2.x1, obj2.y1, obj2.x2, obj2.y2),
                                    (person_obj.x1, person_obj.y1, person_obj.x2, person_obj.y2)):
                        still_hidden = True
                        break

            if still_hidden:
                next_previous_static_hand_events.append(obj2)
            else:
                if obj2.label not in ['person', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']:
                    deletedList.append(obj2)
        
        previous_static_hand_events.clear()
        for obj1 in next_previous_static_hand_events:
            previous_static_hand_events.append(obj1)
    else:
        for obj1 in current_events:
            newObj = RealFinalObject(True, obj1.relative_x, obj1.relative_y,
                                        obj1.center_x, obj1.center_y, obj1.label,
                                        obj1.isTouch, obj1.personId,
                                        obj1.x1, obj1.y1, obj1.x2, obj1.y2, obj1.conf)
            after_frame_datas.append(newObj)
            previous_static_hand_events.append(obj1)
    print(f"next_pre : {next_previous_static_hand_events}")
    print(f"previous : {previous_static_hand_events}")
    # 3. 상태 갱신

            
    '''# # key: (x_quantized, y_quantized, label)
        #print(current_events, previous_static_hand_events)
        current_dict = {make_key(obj): obj for obj in current_events}
        prev_dict = {make_key(obj): obj for obj in previous_static_hand_events}
    # # 새로 생긴 객체 탐색
        for key, cur in current_dict.items():
            if key not in prev_dict:
                after_frame_datas.append(RealFinalObject(True, cur.relative_x, cur.relative_y, cur.center_x, cur.center_y, cur.label, cur.isTouch, cur.personId, cur.x1, cur.y1,cur.x2, cur.y2))

    # # 사라진 객체 탐색
        for key, prev in prev_dict.items():
            if key not in current_dict:
                if is_occluded(prev, 20, Color):
                    continue
                else:
                    after_frame_datas.append(RealFinalObject(False, prev.relative_x, prev.relative_y, prev.center_x, prev.center_y, prev.label, prev.isTouch, prev.personId, prev.x1, prev.y1,prev.x2, prev.y2 ))

    # # 첫 프레임이면 바로 리턴
        if len(previous_static_hand_events) == 0:
            previous_static_hand_events = current_events
            return [RealFinalObject(True, obj.relative_x, obj.relative_y, obj.center_x, obj.center_y, obj.label, obj.isTouch, obj.personId,  obj.x1, obj.y1,obj.x2, obj.y2) for obj in current_events]

        previous_static_hand_events = current_events[:]'''
    return after_frame_datas, deletedList


def all_process_for_detect(pipeline, align, src_points, dst_points, ZWall, previous_static_hand_events, dst):
    frame_color, depth, raw_depth_frame = Get_Frame(pipeline, align)
    # 밝기 증가 (beta 값만 조절)
    '''brightness = 60
    frame_color = cv2.convertScaleAbs(frame_color, alpha=1.0, beta=brightness)'''

    Detected_List,img_for_show = DetectAndGetData(frame_color, src_points, dst)

    #print(f"탐지이후의 객체 수 : {len(Detected_List)}")
    Touched = GetIsTouch(frame_color, depth, Detected_List, ZWall)

    #print(f"터치판정이후의 객체 수 : {len(Touched)}")
    Relative_List=filter_and_print_within_bounds(Touched,src_points,dst_points)
    #print(f"정면뷰 좌표 필터 이후의 객체수 : {len(Relative_List)}")
    Final_Frame_Data_List,deletedList=filter_new_before_StaticAndHand_events(Relative_List,frame_color, previous_static_hand_events)
    #print(f"최종적으로 보낼 객체의 수  : {len(Final_Frame_Data_List)}")

    return Final_Frame_Data_List, img_for_show, deletedList
import random
people_color_list = []
for i in range(100):
    people_color_list.append((random.randint(0,255), random.randint(0,255), random.randint(0,255)))
c = 0
def merge_and_deduplicate(frame, left_Realfinal, right_Realfinal, deletedList1, deletedList2, prev1, prev2):
    
    global c, f, min_confidence_person, min_confidence
    c+=1
    prevMerged = prev1 # + prev2
    for obj in prevMerged:
        x = int(obj.relative_x * 1280)
        y = int(obj.relative_y * 720)
        x1 = int(obj.x1*1280)
        y1 = int(obj.y1*720)
        x2 = int(obj.x2*1280)
        y2 = int(obj.y2*720)
        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,0,0), 2)
        if obj.label not in ['person', 'left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']:
            cv2.putText(frame, f'{obj.label}, {obj.conf:.2f}', (x+30,y), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0), 1, cv2.LINE_AA)
    deletedlistMerged = deletedList1 # + deletedList2
    '''for obj in deletedlistMerged:
        cv2.putText(frame, f'{obj.label} is deleted', (int(obj.relative_x*1280), int(obj.relative_y*720)), cv2.FONT_HERSHEY_SIMPLEX,
            2,(0,0,0), 2, cv2.LINE_AA)'''
    merged = left_Realfinal # + right_Realfinal
    unique = []
    armUnique = []
    personUnique = []
    threshold = 0.1  # 상대 좌표 기준 거리
    from collections import defaultdict
    data = []
    # 1. personId별로 객체를 모으기
    grouped = defaultdict(list)
    for obj in merged:
        if obj.personId is not None:
            grouped[obj.personId].append(obj)
        else:
            data.append(obj)
    # 2. 5개 이상인 personId만 추출

    for personId, objs in grouped.items():
        if len(objs) >= 5:
            data.extend(objs)
    merged = data[:]
    merged.sort(key=lambda x: x.personId if x.personId is not None else -1, reverse=True)

    for idx1, obj1 in enumerate(merged):
        is_duplicate = False

        for obj2 in unique:
            x_diff = obj1.relative_x - obj2.relative_x
            y_diff = obj1.relative_y - obj2.relative_y
            # 👇 label까지 비교
            if obj1.label == obj2.label:
                #threshold = 0.1 if obj1.label == 'person' else 0.2
                thresholdY = 0.03
                thresholdX = 0.021
                if abs(x_diff) < thresholdX and abs(y_diff) < thresholdY:
                    is_duplicate = True
                    break
        if not is_duplicate:
            
            unique.append(obj1)
            # 시각화
            x = int(obj1.relative_x * 1280)
            y = int(obj1.relative_y * 720)
            x1, y1, x2, y2 = map(int, (obj.x1*1280, obj.y1*720, obj.x2*1280, obj.y2*720))
            color = (0,255,255) if obj1.isTouch else (0,0,255)
            if obj1.personId is not None:
                color = people_color_list[obj1.personId]
            if obj1.label == 'person':
                cv2.rectangle(frame, (int(obj1.x1*1280),int(obj1.y1*720)),(int(obj1.x2*1280),int(obj1.y2*720)), color, 5)
            elif obj1.label not in ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']:
                cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
                cv2.putText(frame, f'{obj1.label}, {obj1.conf:.2f}', (x1+30,y1), cv2.FONT_HERSHEY_SIMPLEX, 1,(255,0,0), 1, cv2.LINE_AA)
            #cv2.putText(frame, f'{obj1.label}, {obj1.relative_x:.2f}, {obj1.relative_y:.2f}', (x+30, y), cv2.FONT_HERSHEY_SIMPLEX,
            #2,color, 2, cv2.LINE_AA)
    
    '''for idx1, obj1 in enumerate(unique):
        if obj1.label in ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']:
            isExisted = False
            for idx2, obj2 in enumerate(unique):
                if obj2.label == 'person' and obj2.personId == obj1.personId:
                    isExisted = True
                    break
            if not isExisted:
                for idx3, obj3 in enumerate(merged):
                    x_diff = obj1.relative_x - obj3.relative_x
                    y_diff = obj1.relative_y - obj3.relative_y
                    threshold = 0.2
                    if obj3.label == obj1.label and obj3.personId != obj1.personId:
                        if abs(x_diff) < threshold and abs(y_diff) < threshold:
                            obj1.personId = obj3.personId
    '''
    obj_unique = []
    print("최종 중복 제거된 객체 수 :", len(unique))
    cv2.imshow("asdf", frame)
    key = cv2.waitKey(1)
    if key == ord('i'):
        min_confidence += 0.1
        min_confidence_person += 0.1
        if min_confidence >= 1:
            min_confidence = 1
        if min_confidence_person >= 1:
            min_confidence_person = 1
    elif key == ord('d'):
        min_confidence -= 0.1
        min_confidence_person -= 0.1
        if min_confidence <= 0:
            min_confidence = 0
        if min_confidence_person <= 0:
            min_confidence_person = 0
    for idx1, arm in enumerate(unique):
        if arm.label in ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']:
            for pi in range(idx1+1, len(unique)):
                if arm.personId == unique[pi].personId and arm.personId is not None and arm.label in ['left_elbow', 'right_elbow', 'left_wrist', 'right_wrist']:
                    split1 = unique[pi].label.split("_")
                    split2 = arm.label.split("_")
                    if split1[0] == split2[0] and split1[1] != split2[1]:
                        armUnique.append(FinalArmData(arm.relative_x, arm.relative_y, unique[pi].relative_x, unique[pi].relative_y))
        elif arm.label != 'person':
            obj_unique.append(arm)
    for arm in armUnique:
        cv2.rectangle(frame, (int(arm.x1*1280), int(arm.y1*720)), (int(arm.x2*1280), int(arm.y2*720)), (255,0,0),4)
    save_dir = '2camera_debug'
    save_path = os.path.join(save_dir, f'frame_{c:05d}.png')
    cv2.imwrite(save_path, frame)
    f = frame
    '''for obj in merged:
        print("전체 :",obj.relative_x, obj.relative_y, obj.label)
        key = (round(obj.relative_x, 4), round(obj.relative_y, 4), obj.label)
        if key not in seen:
            seen.add(key)
            unique.append(obj)'''

    return obj_unique, armUnique, deletedlistMerged


print("⚡ 헤더 파일  스캔 완료")

























# pipeline, align = init_camera()
# Color, Depth, raw_depth_frame = Get_Frame(pipeline, align)
# print("초기 프레임 출력 완료")
# Detecte_Marker(Color)
# ZWall = Get_Zwall(raw_depth_frame, src_points)

# _, initial_depth_image, initial_depth_frame = Get_Frame(pipeline, align)
# initial_depth_copy = initial_depth_image.copy()
# start = time.time()
# count = 0
# frame_read_time_avg = 0
# while True:
    
#     before_frame_time = time.time()
#     frame_color, depth, raw_depth_frame = Get_Frame(pipeline, align)
#     after_frame_time = time.time()
#     end = time.time()
#     if end-start > 1:
#         start = time.time()
#         print("-"*50)
#         print(f"현재프레임 : {count}")
#         print(f"평균 읽기 지연 시간 : {frame_read_time_avg/count}")
#         print("-"*50)
#         frame_read_time_avg = 0
#         count = 0
#     count += 1
#     frame_read_time_avg += after_frame_time - before_frame_time
        
#     Detected_List = DetectAndGetData(frame_color, src_points)
#     Touched = GetIsTouch(raw_depth_frame, Detected_List, ZWall)
#     print(f"터치판정이후의 객체 수 : {len(Touched)}")
#     Relative_List=filter_and_print_within_bounds(Touched,src_points,dst_points)
#     print(f"정면뷰 좌표 필터 이후의 객체수 : {len(Relative_List)}")
#     Final_Frame_Data_List=filter_new_before_StaticAndHand_events(Relative_List,frame_color)
#     print(f"최종적으로 보낼 객체의 수  : {len(Final_Frame_Data_List)}")

