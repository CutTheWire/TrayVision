import cv2
import copy
import time
import sys
import ctypes
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import PhotoImage

slider_value = 5
image_blurred_TCD = None

# 윤곽선 사각형으로 다듬는 함수2_기능/CAM 기능 버전/CAM_4.py
def approximate_contour(contour: np.ndarray, epsilon_ratio: float = 0.04) -> np.ndarray:
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx

def remapping_image(frame):
    # 왜곡 계수 설정
    k1, k2, k3 = -0.01, 0.0, 0.0  # 핀큐션 왜곡
    rows, cols = frame.shape[:2]

    # 매핑 배열 생성
    mapy, mapx = np.indices((rows, cols), dtype=np.float32)

    # 중앙점 좌표로 -1~1 정규화 및 극좌표 변환
    mapx = 2 * mapx / (cols - 1) - 1
    mapy = 2 * mapy / (rows - 1) - 1
    r, theta = cv2.cartToPolar(mapx, mapy)

    # 방사 왜곡 변영 연산
    ru = r * (1 + k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6))

    # 직교좌표 및 좌상단 기준으로 복원
    mapx, mapy = cv2.polarToCart(ru, theta)
    mapx = ((mapx + 1) * cols - 1) / 2
    mapy = ((mapy + 1) * rows - 1) / 2

    # 리매핑
    map_image = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    return map_image


def TCDFrame(frame):
    time.sleep(5)
    # 이미지 불러오기
    # frame = cv2.imread('./160728_Origin.jpg')
    # 이미지 크기 조정
    frame = remapping_image(frame)
    frame = cv2.resize(frame, dsize=(640, 480), interpolation=cv2.INTER_AREA)
    frame_origin = copy.deepcopy(frame)
    height, width, _ = frame.shape

    _, binary = on_slider_blurred(frame)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 필터링할 윤곽선 저장 리스트
    filtered_contours = []

    # 윤곽선을 순회하며 면적이 일정 크기 이상인 윤곽선만 저장
    min_contour_area = 0  # 예시로 100 픽셀로 지정
    max_contour_area = 10000000
    for contour in contours:
        area = cv2.contourArea(contour)
        if min_contour_area < area < max_contour_area:
            filtered_contours.append(contour)

    area_dictionary = {}

    # 윤곽선 찾기
    for contour in filtered_contours:
        area = cv2.contourArea(contour)
        if area >= 50:
            area_dictionary[area] = contour
    area_dictionary = dict(sorted(area_dictionary.items(), reverse=True))
    try:
        if list(area_dictionary.values())[0] is not None:
            main_area = list(area_dictionary.values())[0]
    except:
        return frame, frame

    approx_image = approximate_contour(main_area)

    # 사각형 그리기
    if len(approx_image) == 4:  # 꼭지점이 4개일 때만 사각형으로 그리기
        cv2.polylines(frame, [approx_image], True, (0, 0, 255), 2)  # 빨간색으로 윤곽선 그리기

    # 꼭지점 좌표 추출용 배열
    location_nparr = []

    approx = np.array(sorted(approx_image, key=lambda x: x[0][0] + x[0][1]))
    for point in approx:
        x, y = point[0]
        location_nparr.append([x, y])
    location = np.array(location_nparr, np.float32)
    x = int(location[0][0]*0.14); y = int(location[0][1]*0.21); y_1 = int(location[0][1]*0.15)
    location[0][0] += x
    location[2][0] -= x
    
    location[0][1] += y
    location[2][1] += y

    location[1][0] += x
    location[3][0] -= x
    
    location[1][1] -= y_1
    location[3][1] -= y_1

    location2 = np.array([[0, 0], [0, height], [width, 0], [width, height]], np.float32)
    pers = cv2.getPerspectiveTransform(location, location2)

    for i in range(len(location.tolist())):
            output_list.insert(tk.END, f"point {i+1} : {location.tolist()[i]}")
    output_list.insert(tk.END, f"\n\n")
    dst = cv2.warpPerspective(frame_origin, pers, (width, height))

    # 'pers' 행렬의 각 요소를 사용하여 파란색 점 그리기
    for x, y in location:
        cv2.circle(frame, (int(x), int(y)), 5, (255, 0, 0), -1)

    
    # 결과 출력
    cv2.imshow('Contours', frame)
    cv2.imshow('Contouxs', dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def exit_clicked():
    cap.release()
    cv2.destroyAllWindows()
    sys.exit()

def get_monitor_resolution():
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


def cv2_to_tk(image):
    # OpenCV 이미지를 PIL 이미지로 변환
    pil_image = Image.fromarray(image)
    # PIL 이미지를 Tkinter 이미지로 변환
    tk_image = ImageTk.PhotoImage(pil_image)
    return tk_image

def resize_image(image, new_width):
    # 현재 이미지의 크기 가져오기
    height, width, _ = image.shape
    # 새로운 크기 계산
    new_height = int(((new_width / width) * height)*0.9)
    # 크기 조정
    resized_image = cv2.resize(image, (new_width, new_height))
    return resized_image


# 카메라 열기
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # 0은 기본 카메라 장치 번호, 더 많은 카메라가 있는 경우에는 1, 2 등을 시도해볼 수 있습니다.
new_width = 3840
new_height = 2160
cap.set(cv2.CAP_PROP_FRAME_WIDTH, new_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, new_height)

root = tk.Tk()
root.title("MAIN")
root.configure(bg="#666666")
root.state('zoomed')
root.attributes('-fullscreen', True)

# 전체화면 단축키 설정
root.bind("<F11>", lambda event: root.attributes("-fullscreen", not root.attributes("-fullscreen")))
root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))

# 대시보드 프레임 생성
dashboard_frame = tk.Frame(root)
dashboard_frame.configure(bg="#666666")
dashboard_frame.pack(fill="both", expand=True)

button_frame = tk.Frame(dashboard_frame)
button_frame.configure(bg="#666666")
button_frame.pack(fill="both", expand=True)  # 좌측에 배치하고 왼쪽 여백 추가

# 버튼 생성 및 스타일 변경
capture_button = tk.Button(button_frame, text="TCD", command=lambda : TCDFrame(frame), height=3, width=35)
capture_button.place(relx=0.3, rely=0.80, anchor="center")  # 상단 여백과 하단 여백 추가
capture_button.configure(bg='#333333', fg="white", font=("Arial", 14, "bold"))

# 버튼 생성 및 스타일 변경
exit_button = tk.Button(button_frame, text="EXIT", command=exit_clicked, height=3, width=8)
exit_button.place(relx=0.97, rely=0.04, anchor="center")  # 상단 여백과 하단 여백 추가
exit_button.configure(bg='#333333', fg="white", font=("Arial", 10, "bold"))

# 비디오 프레임을 표시할 레이블
video_label = tk.Label(dashboard_frame)
video_label.configure(bg="#666666")
video_label.place(relx=0.5, rely=0.4, anchor="center")

output_list = tk.Listbox(button_frame, height=8, width=45)
output_list.place(relx=0.7, rely=0.85, anchor="center")
output_list.configure(bg='#333333', fg="white", font=("Arial", 14, "bold"))

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output_list.insert(tk.END, f"카메라 해상도: {width}x{height}")

def on_slider_blurred(frame):
    global slider_value
    blur_value = slider_value
    # 블러 적용
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)

    _, binary = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 이미지 크기 조정 및 Tkinter 이미지로 변환
    binary_cv2 = cv2.cvtColor(binary, cv2.COLOR_BGR2RGB)
    binary_resized = Image.fromarray(binary_cv2)
    binary_resized = ImageTk.PhotoImage(binary_resized)
    return binary_resized, binary

def update_button_text(event):
    if slider.get() % 2:
        value = slider.get()
        slider_button.config(text=value)

def button_comand():
    global slider_value
    slider_value = slider_button.cget("text")


slider = tk.Scale(dashboard_frame, from_=1, to=101, orient="horizontal", command = lambda value: update_button_text(value),  length=300, tickinterval=10)
slider.set(5)  # 초기값 설정
slider.place(relx=0.265, rely=0.90, anchor="center")
slider.configure(bg='#333333', fg="white", font=("Arial", 10, "bold"))

# 버튼 생성
slider_button = tk.Button(dashboard_frame, text=0, command= button_comand, height=3, width=13)
slider_button.place(relx=0.395, rely=0.90, anchor="center")
slider_button.configure(bg='#333333', fg="white", font=("Arial", 10, "bold"))

while True:
    # 프레임 읽기
    ret, frame = cap.read()
    if not ret:
        break

    height, width, _ = frame.shape
    cut_width = width // 15
    frame = frame[:, cut_width:width - cut_width]
    frame= resize_image(frame, 1280)

    image_copy = copy.deepcopy(frame)
    image_blurred, _ = on_slider_blurred(image_copy)
    # 이미지 크기 조정
    image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
    image_copy = Image.fromarray(image_copy)
    image_copy = ImageTk.PhotoImage(image_copy)
    
    try:
        video_label.configure(image=image_copy)
        video_label.image = image_copy
        root.update()
    except tk.TclError:
        break

# 카메라 종료
cap.release()
cv2.destroyAllWindows()

# Tkinter 창 실행
root.mainloop()
