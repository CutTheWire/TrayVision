import sys
import cv2
import numpy as np
import copy
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tendo import singleton
from PIL import Image, ImageTk
from pygrabber.dshow_graph import FilterGraph

import image_processing as IP

try:
    me = singleton.SingleInstance()
except:
    sys.exit(1)

# 카메라 탐색
def get_available_cameras() :

    devices = FilterGraph().get_input_devices()

    available_cameras = {}

    for device_index, device_name in enumerate(devices):
        available_cameras[device_index] = device_name

    return available_cameras

camera_names = get_available_cameras()
current_button = ""
threshold = None
frame = None

for i in camera_names:
    if  camera_names[i] == "usb-webcam":
        cap = cv2.VideoCapture(i)
    else:
        cap = cv2.VideoCapture(0)
        
new_width = 2560
new_height = 1440
cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)
root = tk.Tk()
root.title("Dashboard")
root.configure(bg="#666666")
root.state('zoomed')
root.attributes('-fullscreen', True)

#전체화면 단축키 설정
root.bind("<F11>", lambda event: root.attributes("-fullscreen", not root.attributes("-fullscreen")))
root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))

screen_width, screen_height = IP.get_monitor_resolution()
desired_ratio = 16/10
dashboard_ratio = screen_width / screen_height

if dashboard_ratio > desired_ratio:
    dashboard_width = int(screen_height * desired_ratio)-480
    dashboard_height = screen_height-200
else:
    dashboard_height = int(screen_width / desired_ratio)-200
    dashboard_width = screen_width-480


# 대시보드 프레임 생성
dashboard_frame = tk.Frame(root)
dashboard_frame.configure(bg="#666666")
dashboard_frame.pack(fill="both", expand=True)

# 비디오 프레임을 표시할 레이블
video_label = tk.Label(dashboard_frame)
video_label.configure(bg="#666666")
video_label.place(relx=0.44, rely=0.5, anchor="center")  # 좌측에 배치하고 왼쪽 여백 추가

# 콤보박스 선택 시 실행되는 함수
def on_combobox_select(event):
    global threshold
    selected_value = combobox.get()
    threshold.set(float(selected_value))
    tk.DoubleVar().set(float(selected_value))

# 슬라이더 레이블 및 슬라이더
function_frame = tk.Frame(dashboard_frame)
function_frame.configure(bg="#666666")
function_frame.place(relx=0.9, rely=0.5, anchor="center")  # 좌측에 배치하고 왼쪽 여백 추가

threshold_label = tk.Label(function_frame, text="임계값:")
threshold_label.configure(bg="#666666", fg="white", font=("Arial", 12, "bold"))
threshold_label.pack()

threshold = tk.DoubleVar() # 슬라이더 값을 저장하는 변수
threshold.set(46.4) # 초기 슬라이더 값 설정

threshold_slider = tk.Scale(function_frame, from_=0.1, to=80, resolution=0.1, orient=tk.HORIZONTAL, length=200, variable=threshold)
threshold_slider.configure(bg="#666666",  fg="white", font=("Arial", 12, "bold"))
threshold_slider.pack()

# 슬라이더에 포커스를 주고, 키보드 이벤트를 처리하는 함수
def on_slider_focus(event):
    threshold_slider.focus_set(51)

def on_key_press(event):
    if event.keysym == 'Left':
        threshold_slider.set(threshold_slider.get() - 0.1)
    elif event.keysym == 'Right':
        threshold_slider.set(threshold_slider.get() + 0.1)

# 슬라이더에 포커스를 주고, 키보드 이벤트를 처리하는 바인딩
threshold_slider.bind('<FocusIn>', on_slider_focus)
root.bind('<KeyPress>', on_key_press)

# 콤보박스 생성
combobox_frame = tk.Frame(function_frame)
combobox_frame.configure(bg="#666666")
combobox_frame.pack()

none_label = tk.Label(combobox_frame)
none_label.configure(bg="#666666")
none_label.grid(row=0,column=0)

threshold_values = [46.4,47.7,59.0,18.1]
threshold_values.sort()  # 임계값 리스트
combobox = ttk.Combobox(combobox_frame, values=threshold_values, width=11, font="Verdana 16 bold", state="readonly")
combobox.grid(row=1, column=0, rowspan=2)
# 콤보박스 선택 이벤트 바인딩
combobox.bind("<<ComboboxSelected>>", on_combobox_select)

com_bottom_label = tk.Label(combobox_frame, text="\n\n\n임계값 리스트"+
                            "\n —————————— \n"+
                            "10X10\t: 59.0\n"+
                            "5X10\t: 46.4\n"+
                            "S_10x10\t: 18.1\n"+
                            "200-1\t: 47.7\n",
                            fg="white", font=("Arial", 14, "bold"), justify='left')
com_bottom_label.configure(bg="#666666")
com_bottom_label.grid(row=3,column=0)

# 버튼 스타일 변경 함수
def change_button_style(button):
    button.configure(bg='#333333', fg="white", font=("Arial", 12, "bold"))

# 탐지 버튼 클릭 시 실행되는 함수
def capture_button_clicked(button_name):
    global current_button, threshold, frame
    current_button = button_name
    IP.edit_image(frame, current_button, threshold, root)

def clicked_style():
    change_button_style(ten_by_ten_button)
    change_button_style(five_by_ten_button)
    change_button_style(S_ten_by_ten_button)
    change_button_style(two_hundred_button)

#버튼 클릭 시 실행되는 함수
def button_clicked(button_name: str):
    global current_button
    current_button = button_name
    clicked_style()
    capture_button_clicked(button_name)

# 10x10 버튼 생성 및 스타일 변경
ten_by_ten_button = tk.Button(function_frame, text="10x10", command=lambda: button_clicked("10x10"), height=3, width=20)
ten_by_ten_button.pack(pady=(80, 10))  # 상단 여백과 하단 여백 추가

# 5x10 버튼 생성 및 스타일 변경
five_by_ten_button = tk.Button(function_frame, text="5x10", command=lambda: button_clicked("5x10"), height=3, width=20)
five_by_ten_button.pack(pady=(0, 10))  # 하단 여백 추가

#S_10X10 버튼 생성 및 스타일 변경
S_ten_by_ten_button = tk.Button(function_frame, text="S_10x10", command=lambda: button_clicked("S_10x10"), height=3, width=20)
S_ten_by_ten_button.pack(pady=(1, 10))  # 하단 여백 추가

#200-1 버튼 생성 및 스타일 변경
two_hundred_button = tk.Button(function_frame, text="200-1", command=lambda: button_clicked("200-1"), height=3, width=20)
two_hundred_button.pack(pady=(1, 10))  # 상단 여백과 하단 여백 추가

def exit_clicked():
    result = messagebox.askquestion("Exit Confirmation", "프로그램을 종료 하시겠습니까?")
    if result == "yes":
        cap.release()
        cv2.destroyAllWindows()
        root.destroy()
        sys.exit(0)
        sys.exit(1)

# 버튼 생성 및 스타일 변경
exit_button = tk.Button(dashboard_frame, text="X", command=exit_clicked, height=2, width=4)
exit_button.place(relx=0.97, rely=0.04, anchor="center")  # 상단 여백과 하단 여백 추가

# 버튼 스타일 변경
for b in [ten_by_ten_button, five_by_ten_button, S_ten_by_ten_button, two_hundred_button, exit_button]:
    change_button_style(b)

def update_video_stream():
    global frame
    ret, frame = cap.read()
    if ret:
        height, width, _ = frame.shape
        cut_width = width//9
        frame = frame[:,cut_width : width-cut_width]

        frame_c = copy.deepcopy(frame)
        frame_c = cv2.resize(frame_c,(width//3,height//3))
        # OpenCV로 프레임을 읽어와서 이미지 리사이즈 및 변환 처리를 PIL.Image로 수행
        image = cv2.cvtColor(frame_c, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        image = image.resize((dashboard_width, dashboard_height), Image.LANCZOS)  # 더 높은 품질의 리사이징 필터 사용 (예: Image.LANCZOS, Image.BILINEAR)
        image = ImageTk.PhotoImage(image)


        # 레이블에 새 이미지 업데이트
        video_label.configure(image=image)
        video_label.image = image

    # 다음 업데이트를 스케줄링
    root.after(1, update_video_stream)

# 비디오 스트림 업데이트 루프 시작
update_video_stream()

# Tkinter 메인 루프 시작
root.mainloop()

# 자원 해제
cap.release()
cv2.destroyAllWindows()