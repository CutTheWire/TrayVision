import sys
import cv2
from tendo import singleton
import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from screeninfo import get_monitors
import subprocess

camera_device = 0
captured_photo_path = "captured_photo.jpg"
current_button = "10x10"  # 현재 버튼 정보 저장 변수
width, height = 0,0
part_cell_num, empty_cell_num = 0,0
empty_cells, part_cells, part_cells_2 = [],[],[]
allowed_mac_address = "8C-17-59-F1-21-47"  # 허용된 MAC 주소
try:
    me = singleton.SingleInstance()
except:
    sys.exit(1)


def get_physical_address():
    result = subprocess.run(["ipconfig", "/all"], capture_output=True, text=True)
    output = result.stdout
    # 물리적 주소(MAC 주소)를 찾아 반환합니다.
    mac_addresses = re.findall(r"Physical Address[\. ]+: ([\w-]+)", output)
    if mac_addresses:
        return mac_addresses[0].replace("-", ":")  # MAC 주소의 구분자를 ":"로 변경
    else:
        return None

def check_mac_address():
    physical_address = get_physical_address()
    if physical_address == allowed_mac_address:
        return True
    else:
        return False

def capture_photo():
    if not check_mac_address():
        print("허용되지 않은 PC입니다.")
        return

    # 이하 코드 생략

def capture_photo():
    cap = cv2.VideoCapture(camera_device)
    ret, frame = cap.read()
    cv2.imshow('Real-time Video', frame)
    if cv2.waitKey(0) & 0xFF == ord('c'):
        cv2.imwrite(captured_photo_path, frame)
        print("Photo captured!")

        # Perform object detection on captured photo
        perform_object_detection(frame)

    cap.release()

def camera_stream():
    cap = cv2.VideoCapture(camera_device)

    root = tk.Tk()
    root.title("Dashboard")
    root.configure(bg="#666666")
    root.state('zoomed') # 최대화 처리
    root.attributes('-fullscreen', True)
    #전체화면 단축키 설정
    root.bind("<F11>", lambda event: root.attributes("-fullscreen",
                                    not root.attributes("-fullscreen")))
    root.bind("<Escape>", lambda event: root.attributes("-fullscreen", False))

    # Calculate dashboard size based on screen resolution
    monitor = get_monitors()[0]
    screen_width = monitor.width
    screen_height = monitor.height
    desired_ratio = 16/10  # 수정된 가로세로 비율
    dashboard_ratio = screen_width / screen_height

    if dashboard_ratio > desired_ratio:
        dashboard_width = int(screen_height * desired_ratio)-480
        dashboard_height = screen_height-200
    else:
        dashboard_width = screen_width-480
        dashboard_height = int(screen_width / desired_ratio)-200

    def remapping_image(frame):
        # 왜곡 계수 설정
        # k1, k2, k3 = 0.02, 0.0, 0.0 # 배럴 왜곡
        k1, k2, k3 = -0.004, 0.0, 0.0  # 핀큐션 왜곡
        '''
        k1은 주 왜곡 효과를 조절하는 계수 값이 양수일때 이미지가 중심을 기준으로 원형으로 퍼짐, 
        음수일때 이미지가 중심을 기준으로 원형으로 좁아짐
        k2는 k1의 제곱에 비례하는 왜곡 효과
        k3는 k1의 세제곱에 비례하는 왜곡 효과 거의 사용하지 않아 0으로 설정
        '''
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

    # 이미지 윤곽선 처리
    def preprocessing_image(image, n, m):
        # 그레이스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 경계선 감지 (Canny edge detection)
        edges = cv2.Canny(gray, n, m)
        
        # 윤곽선 찾기
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        filtered_contours = []
        # 윤곽선을 순회하며 면적이 일정 크기 이상인 윤곽선만 저장
        if current_button == "S_10x10" or "200-1":
            min_contour_area = 1000 
            max_contour_area = 1000000
            for contour in contours:
                area = cv2.contourArea(contour)
                if area < max_contour_area and area > min_contour_area:
                    filtered_contours.append(contour)
            contours=filtered_contours
        return contours, _

    # 꼭지점 좌표 추출
    def coordinate_extraction(contours):
        for contour in contours:
            # 윤곽선 근사화
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            return approx
    
    # perform_object_detection()에서 이미지 전처리 및 탐색 전 셀 구분선 생성
    def edit_image(image):
        # 이미지 불러오기
        if image is None:
            print("이미지 파일을 찾을 수 없습니다.")
            sys.exit()
        image = remapping_image(image)

        # 이미지 크기 및 중심 좌표 계산
        global width ,height
        height, width, _ = image.shape

        # 초기 n, m 값 설정
        if current_button =="S-10x10":
            n, m = 0, 98
        else:
            n, m= 0, 55

        # 꼭지점 좌표 추출용 배열
        location_nparr = []
        approx = []
        while len(approx) != 4:
            # 이미지 전처리 및 좌표 추출
            contours, _ = preprocessing_image(image, n, m)
            approx = coordinate_extraction(contours)
            if approx is None:
                approx = []
            # n, m 값 조건 확인
            if len(approx) != 4:
                # n, m 값 조건을 충족하지 않을 경우 수정
                n += 1; m += 1
                
                # n, m 값 조건 추가 확인
                if m >= 255:
                    dst = image
                    break
            else:
                # 좌표 정렬
                approx = np.array(sorted(approx, key=lambda x: x[0][0] + x[0][1]))

                # 꼭지점 좌표에 파란 점 찍기
                for point in approx:
                    x, y = point[0]
                    location_nparr.append([x, y+2.0])

                location = np.array(location_nparr, np.float32)
                location2 = np.array([[width, height], [width, 0], [0, height], [0, 0]], np.float32)
                pers = cv2.getPerspectiveTransform(location, location2)
                dst = cv2.warpPerspective(image, pers, (width, height))

                # dst 이미지 방향 보정
                dst_corrected = cv2.transpose(dst)
                dst_corrected = cv2.flip(dst_corrected, 1)

                # 180도 회전
                dst = cv2.rotate(dst_corrected, cv2.ROTATE_90_CLOCKWISE)
        return dst

    # edit_image()로 다듬은 사진의 유닛 갯수와 빈칸 여부를 탐색하는 함수
    def perform_object_detection(frame):
        global width, height
        # 각 칸의 크기 계산
        cell_width = width // 10
        if current_button == "10x10":
            cell_height = height // 10

        elif current_button == "5x10":
            cell_height = height // 5

        elif current_button == "S_10x10":
            cell_height = height // 10

        elif current_button == "200-1":
            cell_height = int((height / 13)-3.4)
            cell_width = int((width / 16)-3)

        def cell_i(i): # cell 좌표 계산 함수
            cell_x = i * cell_width
            cell_y = i * cell_height
            return cell_x,cell_y

        if current_button == "200-1": # 200-1 패널 구분선
            """
            200-1 패널의 상단, 하단, 좌측, 우측 여백의 크기

            상단 : cell_height-10 = 23, 하단 : i=15 ⇒ height-(cell_y*i-10) = 28,
            좌측 : cell_width-16 = 21,  우측 : i=18 ⇒ width-(cell_x*i-16) = 27
            """

            # 구분선 가로
            for i in range(1, 2):   # 상단 빈칸
                _, cell_y = cell_i(i)
                cv2.line(frame, ((cell_width*4)+21, cell_y-10), (width-27, cell_y-10), (205, 0, 205), 1)
            for i in range(2, 14):  # 중앙
                _, cell_y = cell_i(i)
                cv2.line(frame, (21, cell_y-10), (width-27, cell_y-10), (205, 0, 205), 1)
            for i in range(14, 15): # 하단 빈칸
                _, cell_y = cell_i(i)
                cv2.line(frame, (21, cell_y-10), ((width-cell_width*4)-27, cell_y-10), (205, 0, 205), 1)

            # 구분선 세로
            for i in range(1,5):    # 좌측 빈칸
                cell_x, _ = cell_i(i)
                cv2.line(frame, (cell_x-16, cell_height+23), (cell_x-16, height-28), (205, 0, 205), 1)
            for i in range(5,14):   # 중앙
                cell_x, _ = cell_i(i)
                cv2.line(frame, (cell_x-16, 23), (cell_x-16, height-28), (205, 0, 205), 1)
            for i in range(14,18):  # 우측 빈칸
                cell_x, _ = cell_i(i)
                cv2.line(frame, (cell_x-16, 23), (cell_x-16, height-cell_height-28), (205, 0, 205), 1)
            
        else:
            for i in range(1, width//cell_width):
                cell_x, _ = cell_i(i)
                cv2.line(frame, (cell_x , 0), (cell_x , height), (205, 0, 205), 1)

            for i in range(1, height//cell_height):
                _, cell_y = cell_i(i)
                cv2.line(frame, (0, cell_y), (width, cell_y), (205, 0, 205), 1)

        # 이미지의 색상 경계 설정
        lower_color = np.array([0], dtype=np.uint8)     # 검은색 (빈 칸)
        upper_color = np.array([50], dtype=np.uint8)    # 어두운 색상 (부품이 있는 칸)

        # 색상 경계에 따른 이미지 필터링
        mask = cv2.inRange(frame, lower_color, upper_color)

        # 빈 칸과 부품이 있는 칸을 저장할 리스트
        global empty_cells, part_cells, part_cells_2
        empty_cells, part_cells, part_cells_2 = [], [], []
        part_area = []

        print("num : ",cell_width * cell_height * threshold.get()*0.001)
        print("mm__mm")

        def unit_contour(i,j):
            global part_cell_num, empty_cell_num
            if current_button == "200-1":
                cell_y = i * cell_height-12
                cell_x = j * cell_width-18
            else:
                cell_y = i * cell_height
                cell_x = j * cell_width
            
            cell_roi = mask[cell_y : cell_y + cell_height, cell_x:cell_x + cell_width]
            cell_contours, _ = cv2.findContours(cell_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # # 유닛 윤곽선 확인용 
            # contour_color = (0, 0, 255)  # 빨간색 (BGR 순서)
            # cv2.drawContours(frame, cell_contours, -1, contour_color, 1)
            
            if sum(cv2.contourArea(contour) for contour in cell_contours) > 1:
                # 윤곽선 면적 계산
                contour_area = sum(cv2.contourArea(contour) for contour in cell_contours)
                # 부품이 두 개 이상 들어간 칸인지 확인
                num_contours = len(cell_contours)
                # unit_contour_2()에서nu 쓸 모든 유닛이 있는 좌표 저장
                print(contour_area)
                if num_contours == 1:
                    part_cells_2.append((j,i))
                    part_area.append(contour_area)

                # 면적이 임계값을 곱한 값의 이상인지 확인하는 조건문
                if contour_area > cell_width * cell_height * threshold.get()*0.001:
                    part_cells.append((j, i))
                    part_cell_num+=1
                # 서로 떨어져있는 유닛의 면적이 두개 이상인지 확인하는 조건문
                if contour_area <= cell_width * cell_height * threshold.get()*0.001 and num_contours >= 2 and current_button == "10x10":
                    part_cells.append((j, i))
                    part_cell_num+=1

            else:
                empty_cells.append((j, i))
                empty_cell_num+=1

        # 유닛의 면적들 중에 가장 작은 면적을 기준으로 오차 범위 안에 만족하는 셀을 part_cells에 저장하는 함수
        def unit_contour_2():
            global part_cell_num, empty_cell_num
            if current_button == "10x10":   # 10x10 버튼을 눌렀을 때
                min_per = 1.8
                max_per = 3.0
            else:                           # 5x10 버튼을 눌렀을 때
                min_per = 1.85
                max_per = 2.3
            if current_button == "S_10x10" or "200-1":
                pass
            else:
                for i in range(0,len(part_area)):
                        if part_area[i] >= min(part_area)*min_per and part_area[i] <= min(part_area)*max_per:
                            part_cells.append(part_cells_2[i])
                            part_cell_num+=1

        # 윤곽선을 이용하여 빈 칸과 부품이 있는 칸 검출
        if current_button == "200-1":
            for i in range(1,2):
                for j in range(5,17):
                    unit_contour(i,j)
            for i in range(2,13):
                for j in range(1,17):
                    unit_contour(i,j)
            for i in range(13,14):
                for j in range(1,13):
                    unit_contour(i,j)
        else:
            for i in range(10):
                for j in range(10):
                    unit_contour(i,j)
            pass
        
        # unit_contour(,) 특정 셀의 유닛 윤곽선 확인할떄 사용 (# 유닛 윤곽선 확인용)
        # 위 코드를 사용하기 전에 하단에 빨간 사각형 그리기 코드를 주석처리 하고 사용

        # 10x10, 5x10 패널에만 적용 (세워져 있는 부품 2개를 인식하지 못하는 것을 보완)
        if current_button == "10x10" or "5x10":
            unit_contour_2()

        # 결과 이미지를 생성하여 표시
        result_image = remapping_image(frame.copy()) # remapping_image()사용하여 왜곡보정
        
        for cell in empty_cells:
            if current_button == "200-1":
                cell_x = cell[0] * cell_width
                cell_y = cell[1] * cell_height
                # 빨간색 사각형 그리기
                cv2.rectangle(result_image, (cell_x-10, cell_y-5), (cell_x + cell_width-23, cell_y + cell_height-17), (0, 0, 255), 1)
                
            else:
                cell_x = cell[0] * cell_width
                cell_y = cell[1] * cell_height
                # 빨간색 사각형 그리기
                cv2.rectangle(result_image, (cell_x + 15, cell_y + 10), (cell_x + cell_width - 20, cell_y + cell_height - 10), (0, 0, 255), 2)

        # 부품이 있는 칸에 파란색 원 그리기
        for cell in part_cells:
            cell_x = cell[0] * cell_width + cell_width // 2
            cell_y = cell[1] * cell_height + cell_height // 2
            # radius = min(cell_width, cell_height) // 2  # 크기를 더 키워줍니다. 예를 들어 "// 3" 대신 "// 2"로 변경하여 더 큰 원을 그릴 수 있습니다.
            # 현재 cv2.circle(radius=)값을 조절하여 크기 조절 해당코드 미사용
            if current_button == "200-1":
                cv2.circle(result_image, (cell_x-17, cell_y-10), radius=10 , color=(255, 0, 0), thickness=2)  # 파란색 원 그리기
            else:
                cv2.circle(result_image, (cell_x, cell_y), radius=20 , color=(255, 0, 0), thickness=2)  # 파란색 원 그리기
            
        # 이미지 시각화
        fig = plt.figure(figsize=(100, 100))
        plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')

        # 창 닫기 이벤트 처리
        def on_close(event):
            plt.close(fig)

        fig.canvas.mpl_connect('close_event', on_close)
        # Tkinter의 대시보드 창을 생성하여 Figure를 표시
        dashboard_window = tk.Toplevel(root)
        dashboard_window.title("탐지 결과")
        dashboard_window.geometry("+0+0")  # 창의 위치를 왼쪽 상단으로 고정
        dashboard_window.state('zoomed') # 최대화 처리
        dashboard_window.attributes('-fullscreen', True) # 전체화면 처리
        dashboard_window.bind("<Escape>", lambda event: dashboard_window.destroy()) # esc 키 누르면 탐색 결과창 꺼짐
        
        T_threshold=round(cell_width * cell_height * threshold.get()*0.001, 4)
        canvas = FigureCanvasTkAgg(fig, master=dashboard_window)
        canvas.draw()
        canvas.get_tk_widget().pack()
        text_label1 = tk.Label(dashboard_window, text=
                               "임계값\t: "+str(T_threshold)+"\n"+
                               "중복\t: "+str(part_cell_num)+"\n"+
                               "빈곳\t: "+str(empty_cell_num), font=("Arial", 36), justify='left')
        text_label1.place(x=0, y=dashboard_window.winfo_screenheight())
    #____________________ perform_object_detection_end__________________________

    # 버튼 스타일 변경 함수
    def change_button_style(button):
        button.configure(bg='#333333', fg="white", font=("Arial", 12, "bold"))  # 버튼의 배경색, 텍스트 색상 및 폰트 변경

    # 탐지 버튼 클릭 시 실행되는 함수
    def capture_button_clicked():
        ret, frame = cap.read()
        if ret:
            # edit_image()으로 다듬은 사진을 perform_object_detection()사용하여  
            perform_object_detection(edit_image(frame))

    # 10x10 버튼 클릭 시 실행되는 함수
    def ten_by_ten_button_clicked():
        global current_button, part_cell_num, empty_cell_num
        current_button = "10x10"
        part_cell_num, empty_cell_num = 0, 0
        change_button_style(capture_button)
        change_button_style(five_by_ten_button)
        change_button_style(S_ten_by_ten_button)
        change_button_style(two_hundred_button)
        capture_button_clicked()

    # 5x10 버튼 클릭 시 실행되는 함수
    def five_by_ten_button_clicked():
        global current_button, part_cell_num, empty_cell_num
        current_button = "5x10"
        part_cell_num, empty_cell_num = 0, -50
        change_button_style(capture_button)
        change_button_style(five_by_ten_button)
        change_button_style(S_ten_by_ten_button)
        change_button_style(two_hundred_button)
        capture_button_clicked()

    # S_10x10 버튼 클릭 시 실행되는 함수
    def S_ten_by_ten_button_clicked():
        global current_button, part_cell_num, empty_cell_num
        current_button = "S_10x10"
        part_cell_num, empty_cell_num = 0, 0
        change_button_style(capture_button)
        change_button_style(five_by_ten_button)
        change_button_style(S_ten_by_ten_button)
        change_button_style(two_hundred_button)
        capture_button_clicked()

    # 200-1 버튼 클릭 시 실행되는 함수
    def two_hundred_button_clicked():
        global current_button, part_cell_num, empty_cell_num
        current_button = "200-1"
        part_cell_num, empty_cell_num = 0, 0
        change_button_style(capture_button)
        change_button_style(five_by_ten_button)
        change_button_style(S_ten_by_ten_button)
        change_button_style(two_hundred_button)
        capture_button_clicked()

    # 대시보드 프레임 생성
    dashboard_frame = tk.Frame(root)
    dashboard_frame.configure(bg="#666666") 
    dashboard_frame.pack(fill="both", expand=True)

    # 비디오 프레임을 표시할 레이블
    video_label = tk.Label(dashboard_frame)
    video_label.configure(bg="#666666") 
    video_label.pack(side="left", padx=(10, 0))  # 좌측에 배치하고 왼쪽 여백 추가

    # 콤보박스 선택 시 실행되는 함수
    def on_combobox_select(event):
        selected_value = combobox.get()
        threshold.set(float(selected_value))
        tk.DoubleVar().set(float(selected_value))

    # 슬라이더 레이블 및 슬라이더
    slider_frame = tk.Frame(dashboard_frame)
    slider_frame.configure(bg="#666666") 
    slider_frame.pack(side="left", padx=(10, 0))  # 좌측에 배치하고 왼쪽 여백 추가

    threshold_label = tk.Label(slider_frame, text="임계값:")
    threshold_label.configure(bg="#666666", fg="white", font=("Arial", 12, "bold")) 
    threshold_label.pack()

    threshold = tk.DoubleVar() # 슬라이더 값을 저장하는 변수
    threshold.set(46.4) # 초기 슬라이더 값 설정

    threshold_slider = tk.Scale(slider_frame, from_=0.1, to=80, resolution=0.1, orient=tk.HORIZONTAL, length=200,
                                variable=threshold)
    threshold_slider.configure(bg="#666666",  fg="white", font=("Arial", 12, "bold")) 
    threshold_slider.pack()

    # 슬라이더에 포커스를 주고, 키보드 이벤트를 처리하는 함수
    def on_slider_focus(event):
        threshold_slider.focus_set()

    def on_key_press(event):
        if event.keysym == 'Left':
            threshold_slider.set(threshold_slider.get() - 0.1)
        elif event.keysym == 'Right':
            threshold_slider.set(threshold_slider.get() + 0.1)

    # 슬라이더에 포커스를 주고, 키보드 이벤트를 처리하는 바인딩
    threshold_slider.bind('<FocusIn>', on_slider_focus)
    root.bind('<KeyPress>', on_key_press)

    # 콤보박스 생성
    combobox_frame = tk.Frame(slider_frame)
    combobox_frame.configure(bg="#666666")
    combobox_frame.pack()

    none_label = tk.Label(combobox_frame)
    none_label.configure(bg="#666666")
    none_label.grid(row=0,column=0)

    threshold_values = [46.5,35.1,51.5,18.2]
    threshold_values.sort()  # 임계값 리스트
    combobox = ttk.Combobox(combobox_frame, values=threshold_values, width=11, font="Verdana 16 bold", state="readonly")
    combobox.grid(row=1, column=0, rowspan=2)
    # 콤보박스 선택 이벤트 바인딩
    combobox.bind("<<ComboboxSelected>>", on_combobox_select)

    com_bottom_label = tk.Label(combobox_frame, text="\n\n\n임계값 리스트"+
                                "\n —————————— \n"+
                                "10X10\t: 51.5\n"+
                                "5X10\t: 46.5\n"+
                                "S_10x10\t: 18.2\n"+
                                "200-1\t: 35.1\n", fg="white", font=("Arial", 14, "bold"), justify='left')
    com_bottom_label.configure(bg="#666666")
    com_bottom_label.grid(row=3,column=0)

    # 버튼 프레임 생성
    button_frame = tk.Frame(dashboard_frame)
    button_frame.configure(bg="#666666")
    button_frame.pack(side="left", padx=(10, 0))  # 좌측에 배치하고 왼쪽 여백 추가

    # 10x10 버튼 생성 및 스타일 변경
    capture_button = tk.Button(button_frame, text="10x10", command=ten_by_ten_button_clicked, height=3, width=20)
    capture_button.pack(pady=(80, 10))  # 상단 여백과 하단 여백 추가

    # 5x10 버튼 생성 및 스타일 변경
    five_by_ten_button = tk.Button(button_frame, text="5x10", command=five_by_ten_button_clicked, height=3, width=20)
    five_by_ten_button.pack(pady=(0, 10))  # 하단 여백 추가
    
    #S_10X10 버튼 생성 및 스타일 변경
    S_ten_by_ten_button = tk.Button(button_frame, text="S 10x10", command=S_ten_by_ten_button_clicked, height=3, width=20)
    S_ten_by_ten_button.pack(pady=(1, 10))  # 하단 여백 추가

    #200-1 버튼 생성 및 스타일 변경
    two_hundred_button = tk.Button(button_frame, text="200-1", command=two_hundred_button_clicked, height=3, width=20)
    two_hundred_button.pack(pady=(1, 10))  # 상단 여백과 하단 여백 추가

    # 버튼 스타일 변경
    change_button_style(capture_button)
    change_button_style(five_by_ten_button)
    change_button_style(S_ten_by_ten_button)
    change_button_style(two_hundred_button)

    while True:
        ret, frame = cap.read()
        if ret:
            # OpenCV 이미지를 Tkinter 이미지로 변환
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
            try:
                image = ImageTk.PhotoImage(image.resize((dashboard_width, dashboard_height)))
            except(RuntimeError):
                pass

            # 레이블에 이미지 업데이트
            try:
                video_label.configure(image=image)
            except tk.TclError:
                break
            video_label.image = image

        root.update()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Run camera stream
camera_stream()
