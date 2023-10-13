import sys
import cv2
import numpy as np
import copy
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
from PyQt5.QtWidgets import *
import matplotlib
matplotlib.use("TkAgg")

import image_save as IS

current_button = "10x10"
threshold = None
root = None
part_cell_num, empty_cell_num = 0,0
m = 5

def get_monitor_resolution():
    app = QApplication([])
    screen = app.primaryScreen()
    screen_width = screen.size().width()
    screen_height = screen.size().height()
    return screen_width, screen_height

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

    ru = r * (1 + k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6))

    # 직교좌표 및 좌상단 기준으로 복원
    mapx, mapy = cv2.polarToCart(ru, theta)
    mapx = ((mapx + 1) * cols - 1) / 2
    mapy = ((mapy + 1) * rows - 1) / 2

    # 리매핑
    map_image = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
    return map_image

def on_slider_blurred(frame):
    global m
    blur_value = m
    # 블러 적용
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)

    _, binary = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return binary

# 이미지 윤곽선 처리
def preprocessing_image(image: np.ndarray):
    global current_button, threshold
    # 그레이스케일 변환
    binary = on_slider_blurred(image)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []

    # 윤곽선을 순회하며 면적이 일정 크기 이상인 윤곽선만 저장
    min_contour_area = 100000
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_contour_area:
            filtered_contours.append(contour)
    return filtered_contours

# 꼭지점 좌표 추출
def coordinate_extraction(contours):
    for contour in contours:
        # 윤곽선 근사화
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        return approx
    
# perform_object_detection()에서 이미지 전처리 및 탐색 전 셀 구분선 생성
def edit_image(image, current_button_CS, threshold_CS, root_CS):
    global current_button, threshold, root, m

    root = root_CS; current_button = current_button_CS; threshold = threshold_CS

    if image is None:
        print("이미지 파일을 찾을 수 없a니다.")
        sys.exit()
    image = remapping_image(image)

    # 이미지 크기 및 중심 좌표 계산
    global width ,height
    height, width = image.shape[:2]

    # 꼭지점 좌표 추출용 배열
    location_nparr = []
    approx = []
    while len(approx) != 4:
        contours =  preprocessing_image(image)
        approx = coordinate_extraction(contours)

        if approx is None:
            approx = []

        if len(approx) == 4:
            approx = np.array(sorted(approx, key=lambda x: x[0][0] + x[0][1]))
            for point in approx:
                x, y = point[0]
                location_nparr.append([x, y+2.0])
            location = np.array(location_nparr, np.float32)
            print(location)

            if current_button == "200-1":
                x = int(location[0][0]*0.84); x_1 = int(location[0][0]*0.6); y = int(location[0][1]*0.91); y_1 = int(location[0][1]*0.6)
                location[0][0] += x_1
                location[2][0] -= x

                location[0][1] += y
                location[2][1] += y

                location[1][0] += x_1
                location[3][0] -= x

                location[1][1] -= y_1
                location[3][1] -= y_1

            elif current_button == "S_10x10":
                x = int(location[0][0]*0.71); x_1 = int(location[0][0]*0.71); y = int(location[0][1]*0.85); y_1 = int(location[0][1]*0.5)
                location[0][0] += x_1
                location[2][0] -= x

                location[0][1] += y
                location[2][1] += y

                location[1][0] += x_1
                location[3][0] -= x

                location[1][1] -= y_1
                location[3][1] -= y_1
            
            elif current_button == "10x10":
                x = int(location[0][0]*0.05); x_1 = int(location[0][0]*0.05); y = int(location[0][1]*0.05); y_1 = int(location[0][1]*0.05)
                location[0][0] += x_1
                location[2][0] -= x

                location[0][1] += y
                location[2][1] += y

                location[1][0] += x_1
                location[3][0] -= x

                location[1][1] -= y_1
                location[3][1] -= y_1
            
            elif current_button == "5x10":
                x = int(location[0][0]*0.15); x_1 = int(location[0][0]*0.06); y = int(location[0][1]*0.05); y_1 = int(location[0][1]*0.1)
                location[0][0] += x_1
                location[2][0] -= x

                location[0][1] += y
                location[2][1] += y

                location[1][0] += x_1
                location[3][0] -= x

                location[1][1] -= y_1
                location[3][1] -= y_1
            
            
            location2 = np.array([[0, 0], [0, height], [width, 0], [width, height]], np.float32)
            pers = cv2.getPerspectiveTransform(location, location2)
            dst = cv2.warpPerspective(image, pers, (width, height))
            dst = cv2.resize(dst,(int(height*1.3),height))

        else:
            m += 2
            if m == 101:
                dst = image
                dst = cv2.resize(dst,(int(height*1.3),height))
                break
        IS.image_save(dst, "Origin", current_button)
    return perform_object_detection(dst)

# edit_image()로 다듬은 사진의 유닛 갯수와 빈칸 여부를 탐색하는 함수
def perform_object_detection(frame):
    global part_cell_num, empty_cell_num
    height, width = frame.shape[:2]
    # 각 칸의 크기 계산
    cell_width = width // 10
    if current_button == "10x10":
        cell_height = height // 10

    elif current_button == "5x10":
        cell_height = height // 5

    elif current_button == "S_10x10":
        cell_height = height // 10

    elif current_button == "200-1":
        cell_height = int((height / 13))
        cell_width = int((width / 16))

    def cell_i(i): # cell 좌표 계산 함수
        cell_x = i * cell_width
        cell_y = i * cell_height
        return cell_x,cell_y

    def draw_line(frame,
                    start_x, start_y,
                    end_x, end_y):
        cv2.line(frame,
                    (start_x, start_y),
                    (end_x, end_y),
                    (205, 0, 205), 1)

    if current_button == "200-1":
        # 구분선 가로와 세로
        for i in range(1, 18):
            cell_x, cell_y = cell_i(i)
            if  i <= 15:
                draw_line(frame, cell_x, 0, cell_x, height)
                
            if i <= 12:
                draw_line(frame, 0, cell_y, width, cell_y)
            
    else:
        for i in range(1, width // cell_width):
            cell_x = i * cell_width
            draw_line(frame, cell_x, 0, cell_x, height)

        for i in range(1, height // cell_height):
            cell_y = i * cell_height
            draw_line(frame, 0, cell_y, width, cell_y)

    if current_button == "10x10" or "5x10":
        C1_frame = copy.deepcopy(frame)
        # 이미지의 색상 경계 설정
        lower_color = np.array([0], dtype=np.uint8)     # 검은색 (빈 칸)
        upper_color = np.array([70], dtype=np.uint8)    # 어두운 색상 (부품이 있는 칸)

        brightness_factor = 1  # 밝기를 증가시킬 비율 (1.0보다 크면 밝아지고, 1.0보다 작으면 어두워집니다.)
        brightened_frame = cv2.convertScaleAbs(C1_frame, alpha=brightness_factor, beta=0)

        mask = cv2.inRange(brightened_frame, lower_color, upper_color)

    elif current_button == "200-1" or "S_10x10":
        C2_frame = copy.deepcopy(frame)
        # Noise removal
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        bin_img = cv2.morphologyEx(C2_frame, cv2.MORPH_OPEN, kernel)

        # Image grayscale conversion and contrast adjustment
        bin_img = np.clip((1 + 2.3) * bin_img - 128 * 2.3, 0, 255).astype(np.uint8)

        gray_img = cv2.cvtColor(bin_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        ret, binary_img = cv2.threshold(gray_img,  245 , 255, cv2.THRESH_BINARY_INV)
        inverted_binary_img = cv2.bitwise_not(binary_img)
        mask = cv2.inRange(inverted_binary_img, np.array([0], dtype=np.uint8), np.array([1], dtype=np.uint8))

    # 빈 칸과 부품이 있는 칸을 저장할 리스트
    empty_cells, part_cells, contour_cell = [], [], []

    def unit_contour(i,j):
        global part_cell_num, empty_cell_num
        cell_roi, cell_contours, contour_area = None, None, None
        cell_y = i * cell_height
        cell_x = j * cell_width
        
        cell_roi = mask[cell_y : cell_y + cell_height, cell_x:cell_x + cell_width]
        cell_contours, _ = cv2.findContours(cell_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 윤곽선 면적 계산
        contour_area = sum(cv2.contourArea(contour) for contour in cell_contours)

        if contour_area > 1:

        
        if sum(cv2.contourArea(contour) for contour in cell_contours) > 1:
            # 윤곽선 면적 계산
            contour_area = sum(cv2.contourArea(contour) for contour in cell_contours)
            # 부품이 두 개 이상 들어간 칸인지 확인
            num_contours = len(cell_contours)
            contour_cell.append(num_contours)

            # print(contour_area) # 유닛 면적 확인용
            # 면적이 임계값을 곱한 값의 이상인지 확인하는 조건문
            if contour_area > cell_width * cell_height * threshold.get()*0.001:
                    part_cells.append((j, i))
                    part_cell_num+=1
        else:
            empty_cells.append((j, i))
            empty_cell_num+=1

    # 윤곽선을 이용하여 빈 칸과 부품이 있는 칸 검출
    if current_button == "200-1":
        for i in range(0, 13):
            for j in range(0, 17):
                if i == 0 and j not in range(4, 17):
                    continue
                if i == 12 and j not in range(0, 12):
                    continue
                unit_contour(i, j)
    elif current_button == "5x10":
        for i in range(5):
            for j in range(10):
                unit_contour(i, j)
    else:
        for i in range(10):
            for j in range(10):
                unit_contour(i, j)


    # 결과 이미지를 생성하여 표시
    result_image = copy.deepcopy(frame)
    for cell in empty_cells:
        cell_x = cell[0] * cell_width
        cell_y = cell[1] * cell_height
        # 빨간색 사각형 그리기
        cv2.rectangle(result_image,
                        (cell_x + 15, cell_y + 10),
                        (cell_x + cell_width - 20,
                        cell_y + cell_height - 10),
                        (0, 0, 255), 2)

    # 부품이 있는 칸에 파란색 원 그리기
    for cell in part_cells:
        cell_x = cell[0] * cell_width + cell_width // 2
        cell_y = cell[1] * cell_height + cell_height // 2
        '''
        radius = min(cell_width, cell_height) // 3  # 크기를 더 키워줍니다.
        예를 들어 "// 3" 대신 "// 2"로 변경하여 더 큰 원을 그릴 수 있습니다.
        '''
        cv2.circle(result_image, (cell_x, cell_y),
                        radius=cell_width//3,
                        color=(255, 0, 0),
                        thickness=2)  # 파란색 원 그리기

    # 이미지 시각화
    IS.image_save(result_image, "Scan", current_button)

    def close_dashboard(event):
        dashboard_window.destroy()

    # Tkinter의 대시보드 창을 생성하여 Figure를 표시
    dashboard_window = tk.Toplevel(root)
    dashboard_window.title("탐지 결과")
    dashboard_window.geometry("+0+0")  # 창의 위치를 왼쪽 상단으로 고정
    dashboard_window.state('zoomed') # 최대화 처리
    dashboard_window.bind("<Escape>", close_dashboard) # esc 키 누르면 탐색 결과창 꺼짐

    screen_width, screen_heigth = get_monitor_resolution()

    height, width = result_image.shape[:2]

    re_width = int(screen_width//2)
    re_height = int(screen_heigth//2)

    result_image = cv2.resize(result_image,(re_width, re_height))
    # PIL 이미지로 변환
    pil_image = Image.fromarray(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    # Tkinter 이미지로 변환
    tk_image = ImageTk.PhotoImage(image=pil_image)
    
    # 이미지 표시
    image_label = ttk.Label(dashboard_window,
                            image=tk_image)
    image_label.image = tk_image
    image_label.place(relx= 0.5, rely=0.5, anchor="center")

    if current_button == "200-1":
        empty_cell_num -= 12
    

    T_threshold=round(cell_width * cell_height * threshold.get()*0.001, 4)
    text_label1 = tk.Label(dashboard_window, text=
                            "임계값\t: "+str(T_threshold)+"\n"+
                            "중복\t: "+str(part_cell_num)+"\n"+
                            "빈곳\t: "+str(empty_cell_num),
                            font=("Arial", 24), justify='left')
    text_label1.place(x=20, y=dashboard_window.winfo_screenheight()//2)
