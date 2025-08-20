import cv2
import numpy as np
import pyrealsense2 as rs
import threading
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
import os
import sys
sys.path.append("C:/Users/smpi9/Downloads/yolo_project/ultralytics")
from ultralytics import YOLO

model = YOLO('C:/Users/smpi9/Downloads/yolo_project/ultralytics/runs/detect/best3(99).pt')

wall_width = 300  # 테스트할 벽의 가로
wall_height = 210  # 테스트할 벽의 세로

projector_width = 3840
projector_height = 2160

half_wall_width = wall_width / 2

left_dst_points = np.float32([
    [0, 0],
    [wall_width, 0],
    [wall_width, wall_height],
    [0, wall_height]
])
right_dst_points = np.float32([
    [138, 0],
    [wall_width, 0],
    [wall_width, wall_height],
    [138, wall_height]
])
left_src_points = np.float32([
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0]
])
right_src_points = np.float32([
    [0, 0],
    [0, 0],
    [0, 0],
    [0, 0]
])

Per_Frame = 5  # 1초에 몇 프레임 정도 사진을 찍나
# RealSense D455 Exposure/Gain 범위
EXPOSURE_MIN, EXPOSURE_MAX = 41, 10000
GAIN_MIN, GAIN_MAX = 0, 128

# 시리얼 번호
LEFT_SERIAL = '339522300522'
RIGHT_SERIAL = '246322300435'

# 스레드 동기화
lock = threading.Lock()
stop_event = threading.Event()

pipelines = {}
sensors = {}

SETTING_FILE = "camera_setting_value.txt"


def save_settings_to_file():
    """현재 입력창 값과 마커 좌표를 파일에 저장"""
    values = [
        entry_vars["left_exp"].get(),
        entry_vars["left_gain"].get(),
        entry_vars["right_exp"].get(),
        entry_vars["right_gain"].get()
    ]

    def points_to_str(points):
        if points is None or len(points) != 4:
            return "0,0 0,0 0,0 0,0"
        return " ".join([f"{int(x)},{int(y)}" for x, y in points])

    with open(SETTING_FILE, "w") as f:
        f.write(" ".join(values) + "\n")
        f.write(points_to_str(left_src_points) + "\n")
        f.write(points_to_str(right_src_points) + "\n")


def init_cam(exposure, gain, serial):
    pipe = rs.pipeline()
    config = rs.config()
    config.enable_device(serial)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
    profile = pipe.start(config)

    rgb_sensor = profile.get_device().query_sensors()[1]
    rgb_sensor.set_option(rs.option.enable_auto_exposure, False)
    rgb_sensor.set_option(rs.option.exposure, exposure)
    rgb_sensor.set_option(rs.option.gain, gain)

    return pipe, rgb_sensor


def get_recommended_exposure_gain(img_gray):
    mean_brightness = np.mean(img_gray)

    return mean_brightness




def show_frames_tk():
    with lock:
        frames_left = pipelines["left"].wait_for_frames()
        frames_right = pipelines["right"].wait_for_frames()

        img_left = np.asanyarray(frames_left.get_color_frame().get_data())
        img_right = np.asanyarray(frames_right.get_color_frame().get_data())

    # --- 모델 추론 및 결과 표시 ---
    def draw_yolo_results(img, model):
        results = model(img)[0]
        for box in results.boxes:
            xyxy = box.xyxy[0].cpu().numpy().astype(int)
            conf = box.conf[0].cpu().numpy()
            cls = int(box.cls[0].cpu().numpy())
            class_name = model.names[cls] if hasattr(model, 'names') else str(cls)
            cv2.rectangle(img, (xyxy[0], xyxy[1]), (xyxy[2], xyxy[3]), (0, 255, 0), 2)
            label = f"{class_name} {conf:.2f}"
            cv2.putText(img, label, (xyxy[0], xyxy[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        return img

    img_left = draw_yolo_results(img_left, model)
    img_right = draw_yolo_results(img_right, model)

    img_left_resized = cv2.resize(img_left, (640, 360))
    img_right_resized = cv2.resize(img_right, (640, 360))
    combined = np.hstack((img_left_resized, img_right_resized))

    combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(combined_rgb)
    img_tk = ImageTk.PhotoImage(image=img_pil)

    video_label.imgtk = img_tk
    video_label.configure(image=img_tk)

    # 밝기 분석 및 추천 노출/게인 계산
    img_gray = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    mean_bright = get_recommended_exposure_gain(img_gray)
    img_gray2 = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    mean_bright2 = get_recommended_exposure_gain(img_gray2)

    recommendation_text = (
        f"왼쪽 현재 평균 밝기: {mean_bright:.1f}\n"
        f"오른쪽 현재 평균 밝기: {mean_bright2:.1f}\n"
        f"추천 밝기: 80~85\n"
        "적용하려면 해당 값을 입력 후 '반영' 버튼 클릭"
    )
    recommend_label.config(text=recommendation_text)

    if not stop_event.is_set():
        root.after(10, show_frames_tk)


def apply_manual_settings(side):
    try:
        e = int(entry_vars[side + '_exp'].get())
        g = int(entry_vars[side + '_gain'].get())
    except ValueError:
        messagebox.showerror("입력 오류", "숫자만 입력하세요!")
        return

    if not (EXPOSURE_MIN <= e <= EXPOSURE_MAX and GAIN_MIN <= g <= GAIN_MAX):
        messagebox.showerror("범위 오류", f"Exposure: {EXPOSURE_MIN}~{EXPOSURE_MAX}, Gain: {GAIN_MIN}~{GAIN_MAX}")
        return

    serial = LEFT_SERIAL if side == "left" else RIGHT_SERIAL

    with lock:
        pipelines[side].stop()
        pipelines[side], sensors[side] = init_cam(e, g, serial)

    save_settings_to_file()


def set_entries_state(side, state):
    exp_entry[side].config(state=state)
    gain_entry[side].config(state=state)


def get_4_points_from_click(img, window_name="Click 4 Points"):
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
                cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
                cv2.imshow(window_name, param)

    clone = img.copy()
    cv2.imshow(window_name, clone)
    cv2.setMouseCallback(window_name, mouse_callback, clone)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            points.clear()
            break
        if len(points) == 4:
            break
    cv2.destroyWindow(window_name)
    return points


def draw_markers_and_save():
    global left_src_points, right_src_points

    with lock:
        frames_left = pipelines["left"].wait_for_frames()
        frames_right = pipelines["right"].wait_for_frames()

        img_left = np.asanyarray(frames_left.get_color_frame().get_data())
        img_right = np.asanyarray(frames_right.get_color_frame().get_data())

    left_points = get_4_points_from_click(img_left, "Left Camera: Click 4 Points")
    if len(left_points) != 4:
        messagebox.showerror("에러", "왼쪽 카메라에서 4개의 점을 모두 찍어야 합니다.")
        return

    right_points = get_4_points_from_click(img_right, "Right Camera: Click 4 Points")
    if len(right_points) != 4:
        messagebox.showerror("에러", "오른쪽 카메라에서 4개의 점을 모두 찍어야 합니다.")
        return

    left_src_points = np.float32(left_points)
    right_src_points = np.float32(right_points)

    save_settings_to_file()
    messagebox.showinfo("완료", "마커 좌표가 저장되었습니다.")


# 초기 카메라 시작 (Manual 모드, 기본값 1000, 64)
pipelines["left"], sensors["left"] = init_cam(1000, 64, LEFT_SERIAL)
pipelines["right"], sensors["right"] = init_cam(1000, 64, RIGHT_SERIAL)

# GUI
root = tk.Tk()
root.title("RealSense D455 Camera Controller")

entry_vars = {
    "left_exp": tk.StringVar(value="1000"),
    "left_gain": tk.StringVar(value="64"),
    "right_exp": tk.StringVar(value="1000"),
    "right_gain": tk.StringVar(value="64"),
}

exp_entry = {}
gain_entry = {}

guide_text = (
    "Exposure와 Gain 설정 가이드:\n"
    "- Exposure: 41(최소) ~ 10000(최대) 사이 값 입력\n"
    "  * 낮은 값은 밝기 어둡게, 높은 값은 밝게 설정\n"
    "- Gain: 0 ~ 128 사이 값 입력\n"
    "  * Gain은 이미지 밝기 증폭, 너무 높이면 노이즈 증가\n"
    "- 적절한 설정은 조명 환경에 따라 다름\n"
    "  * 밝은 환경: Exposure 낮게, Gain 낮게 설정\n"
    "  * 어두운 환경: Exposure와 Gain을 적절히 올려 사용\n"
)

guide_label = tk.Label(root, text=guide_text, justify="left", fg="blue")
guide_label.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="w")

recommend_label = tk.Label(root, text="", justify="left", fg="green")
recommend_label.grid(row=1, column=0, columnspan=2, padx=10, pady=(0,10), sticky="w")

# 카메라 설정 프레임 (row=2)
for idx, side in enumerate(["left", "right"]):
    frame = tk.LabelFrame(root, text=f"{side.capitalize()} Camera", padx=5, pady=5)
    frame.grid(row=2, column=idx, padx=10, pady=10)

    tk.Label(frame, text="Exposure").grid(row=0, column=0)
    exp_entry[side] = tk.Entry(frame, textvariable=entry_vars[side + '_exp'], width=10)
    exp_entry[side].grid(row=0, column=1)

    tk.Label(frame, text="Gain").grid(row=1, column=0)
    gain_entry[side] = tk.Entry(frame, textvariable=entry_vars[side + '_gain'], width=10)
    gain_entry[side].grid(row=1, column=1)

    tk.Button(frame, text="반영", command=lambda s=side: apply_manual_settings(s)).grid(row=2, column=0, columnspan=2, pady=5)

    set_entries_state(side, "normal")  # 항상 입력 가능하게


# 영상 표시 영역 (row=3)
video_label = tk.Label(root)
video_label.grid(row=3, column=0, columnspan=2)

# 마커 그리기 버튼 추가 (row=4)
marker_btn = tk.Button(root, text="마커 그리기", command=draw_markers_and_save)
marker_btn.grid(row=4, column=0, columnspan=2, pady=10)

show_frames_tk()

root.protocol("WM_DELETE_WINDOW", lambda: (stop_event.set(), root.destroy()))
root.mainloop()

stop_event.set()
for p in pipelines.values():
    p.stop()
