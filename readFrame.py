import cv2
import os
import time
import pyrealsense2 as rs
import numpy as np

save_dir = 'new_frame15'
os.makedirs(save_dir, exist_ok=True)

frame_count = 0
saved_count = 0

print("🟡 1초에 1장 저장, 화질 우선")
print("ESC: 종료")

def init_camera():
    pipeline = rs.pipeline()
    config = rs.config()
    ctx = rs.context()
    devices = ctx.query_devices()
    if not devices:
        raise RuntimeError("❌ 연결된 RealSense 장치가 없습니다.")
    serial = devices[0].get_info(rs.camera_info.serial_number)
    print(f"✅ 연결된 장치 시리얼: {serial}")
    config.enable_device(serial)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 15)
    config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 15)
    profile = pipeline.start(config)

    # ▣ 수동 노출 설정 (화질 최우선)
    rgb_sensor = profile.get_device().query_sensors()[1]  # [1] = RGB sensor
    rgb_sensor.set_option(rs.option.enable_auto_exposure, False)
    rgb_sensor.set_option(rs.option.exposure, 1000)  # 최대 밝게 (가능하면 800~1600 사이 실험)
    rgb_sensor.set_option(rs.option.gain, 64)       # 밝기 강화, 노이즈가 적당하면 200까지도 OK

    align = rs.align(rs.stream.color)
    return pipeline, align

def Get_Frame(pipeline, align):
    frames = pipeline.wait_for_frames()
    aligned_frames = align.process(frames)
    color = aligned_frames.get_color_frame()
    return np.asanyarray(color.get_data())

pipeline, align = init_camera()
save_interval = 1.0  # 1초 간격 저장
last_saved = time.time()

while True:
    frame = Get_Frame(pipeline, align)
    cv2.imshow("frame", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

    now = time.time()
    if now - last_saved >= save_interval:
        save_path = os.path.join(save_dir, f'{save_dir}_{frame_count:05d}.png')
        cv2.imwrite(save_path, frame, [cv2.IMWRITE_PNG_COMPRESSION, 0])  # 무압축 PNG
        print(f'📸 저장: {save_path}')
        last_saved = now
        frame_count += 1

cv2.destroyAllWindows()
pipeline.stop()
