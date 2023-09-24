import cv2
import tkinter as tk
from tkinter import ttk

# 카메라 객체 생성
# 0은 기본 카메라를 의미, 여러 카메라가 연결되어 있을 경우 숫자를 변경하여 선택할 수 있습니다.
cap = cv2.VideoCapture(0)
button_C = ""

def is_integer(value: str):
    global button_C
    try:
        int(value)
        if int(value)>=2:
            return False
        else:
            return True
        
    except ValueError:
        return False

def set_frame_rate(value: str):
    global button_C
    if button_C == "EXPOSURE":
        if is_integer(value):
            EXPOSURE_value = int(value)
            cap.set(cv2.CAP_PROP_EXPOSURE, EXPOSURE_value)
        else:
            print("올바른 프레임 속도 값을 입력하세요. 2 미만의 정수만 입력 가능합니다.")
    elif button_C == "gain":
        gain_value = float(value)
        cap.set(cv2.CAP_PROP_GAIN, gain_value)
    elif button_C == "FPS":
        FPS_value = float(value)
        cap.set(cv2.CAP_PROP_FPS, FPS_value)

    print(f"노출: {cap.get(cv2.CAP_PROP_EXPOSURE)} ")
    print(f"게인: {cap.get(cv2.CAP_PROP_GAIN)} ")
    print(f"프레임: {cap.get(cv2.CAP_PROP_FPS)} ")

def main():
    
    if not cap.isOpened():
        print("카메라를 열 수 없습니다.")
        return
    print(type(cv2.CAP_PROP_EXPOSURE))
    
    # 새로운 해상도 설정
    new_width = 9000
    new_height = 9000

    # 카메라 기능 변경
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)
    cap.set(cv2.CAP_PROP_EXPOSURE, -5)

    # 카메라의 가능한 해상도와 프레임 확인
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    shutter = int(cap.get(cv2.CAP_PROP_EXPOSURE))

    print(f"카메라 해상도: {width}x{height}")
    print(f"셔터 속도: {shutter} ")

    root = tk.Tk()
    root.title("Camera Frame Rate Control")
    main_frame = tk.Frame(root)
    main_frame.pack(fill="both", expand=True)

# EXPOSURE
    def apply_EXPOSURE_rate():
        global button_C
        button_C = "EXPOSURE"
        return set_frame_rate(EXPOSURE_entry.get())
    
    EXPOSURE_frame=tk.Frame(main_frame)
    EXPOSURE_frame.pack()

    EXPOSURE_label = tk.Label(EXPOSURE_frame, text=" 노출 값 :", font=("Arial", 12))
    EXPOSURE_label.pack(side=tk.TOP)

    EXPOSURE_entry = tk.Entry(EXPOSURE_frame, font=("Arial", 12))
    EXPOSURE_entry.pack()

    EXPOSURE_apply_button = ttk.Button(EXPOSURE_frame, text="Apply", command=apply_EXPOSURE_rate)
    EXPOSURE_apply_button.pack(side=tk.BOTTOM)

# GAIN
    def apply_gain_rate():
        global button_C
        button_C = "gain"
        return set_frame_rate(gain_entry.get())
    
    gain_frame=tk.Frame(main_frame)
    gain_frame.pack()

    gain_label = tk.Label(gain_frame, text=" 게인 값 :", font=("Arial", 12))
    gain_label.pack(side=tk.TOP)

    gain_entry = tk.Entry(gain_frame, font=("Arial", 12))
    gain_entry.pack()

    apply_button = ttk.Button(gain_frame, text="Apply", command=apply_gain_rate)
    apply_button.pack(side=tk.BOTTOM)

# FPS
    def apply_FPS_rate():
        global button_C
        button_C = "FPS"
        return set_frame_rate(FPS_entry.get())
    
    FPS_frame=tk.Frame(main_frame)
    FPS_frame.pack()

    FPS_label = tk.Label(FPS_frame, text=" 프레임 값 :", font=("Arial", 12))
    FPS_label.pack(side=tk.TOP)

    FPS_entry = tk.Entry(FPS_frame, font=("Arial", 12))
    FPS_entry.pack()

    apply_button = ttk.Button(FPS_frame, text="Apply", command=apply_FPS_rate)
    apply_button.pack(side=tk.BOTTOM)

# CAMERA
    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.moveWindow("Camera", 0, 0)  # 창을 1번 모니터로 이동
    cv2.setWindowProperty("Camera", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    # 카메라 영상 캡처 및 화면에 출력
    while True:
        ret, frame = cap.read()

        if not ret:
            print("영상을 읽을 수 없습니다.")
            break

        cv2.imshow("Camera", frame)

        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # 'Esc' key
            break

        # GUI 업데이트
        root.update()

    # 해제
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
