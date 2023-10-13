#Ver.23.10.
import sys
import cv2
import os
import numpy as np
import copy
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
from tkinter import font
from PIL import Image, ImageTk
from PyQt5.QtWidgets import QApplication
import matplotlib.pyplot as plt
from screeninfo import get_monitors

import bin.logo_data as logo_image
import bin.set_data as set_image

import IMG.image_processing as IP
from IMG.IPP import ImageCV
from IMG.camera import Camera
from IMG.Image_Save import save

from TW.TWSM import TW
from TW.Loading import LoadingScreen

from TW.setting_view import SettingsWindow

class MainView:
    def __init__(self, root):
        self.label_style ={
            'bg': '#232326',
            'fg': 'white',
            'font': font.Font(family="Helvetica", size=15) }
        
        self.textbox_style ={
            'bg': '#232326',
            'fg': 'white',
            'font': font.Font(family="Helvetica", size=25) }
        
        self.text_label_style ={
            'bg': '#343437',
            'fg': 'white',
            'font': font.Font(family="Helvetica", size=15) }
        
        self.exit_label_style ={
            'bg': '#B43437',
            'fg': 'white',
            'font': font.Font(family="Helvetica", size=15) }

        self.back_style ={
            'bg' : '#343437' }
        
        self.root_style ={
            'bg' : '#565659' }
        
        self.threshold = tk.DoubleVar() # 슬라이더 값을 저장하는 변수
        self.threshold.set(200.0)
        self.threshold_values = {}
        L_image = tk.PhotoImage(data=logo_image.image_base64)
        S_image = tk.PhotoImage(data=set_image.image_base64)

        # root 설정
        self.root = root
        self.root.title("Micro_TWCV")
        self.root.configure(self.root_style)
        self.root.state('zoomed')
        self.root.attributes('-fullscreen', True)
        
        monitors = get_monitors()
        self.screen_width, self.screen_height = monitors[0].width, monitors[0].height
        self.frame_height = self.screen_height // 20

        # 그리드의 크기를 설정합니다.
        self.root.grid_rowconfigure((0,2,3,4), weight=1)
        self.root.grid_rowconfigure(1, weight=15)
        self.root.grid_columnconfigure(0, weight=10)
        # Frame 생성
        self.frame1 = tk.Frame(self.root, width=self.screen_width, height = self.frame_height)
        self.frame2_1 = tk.Frame(self.root, width=self.screen_width, height = (self.screen_height - self.frame_height*4))
        self.frame2_2 = tk.Frame(self.root, width=self.screen_width, height = (self.screen_height - self.frame_height*4))
        self.frame3 = tk.Frame(self.root, width=self.screen_width, height = self.frame_height)
        self.frame4 = tk.Frame(self.root, width=self.screen_width, height = self.frame_height)
        self.frame5 = tk.Frame(self.root, width=self.screen_width, height = self.frame_height)
        # Frame 배치
        self.frame1.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="new")
        self.frame2_1.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        self.frame2_2.grid(row=1, column=1, padx=10, pady=10, sticky="nsew")
        self.frame3.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="sew")
        self.frame4.grid(row=3, column=0, columnspan=2, padx=10, pady=10, sticky="sew")
        self.frame5.grid(row=4, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # 그리드의 크기를 설정합니다
        self.frame3.columnconfigure((0, 1, 2), weight=0)
        self.frame3.columnconfigure(2, weight= 1)
        self.frame4.columnconfigure((0, 1, 2), weight=0)
        self.frame4.columnconfigure(2, weight= 1)
        # Lable 생성
        self.Label2_1 = tk.Label(self.frame2_1)
        self.Label2_2 = tk.Label(self.frame2_2)
        self.Label3_1 = tk.Label(self.frame3)
        self.Label3_2 = tk.Label(self.frame3)
        self.Label3_3 = tk.Label(self.frame3)
        self.Label4_1 = tk.Label(self.frame4)
        self.Label4_2 = tk.Label(self.frame4)
        self.Label4_3 = tk.Label(self.frame4)
        # Lable 배치
        self.Label2_1.pack(fill=tk.BOTH, expand=True)
        self.Label2_2.pack(fill=tk.BOTH, expand=True)
        self.Label3_1.grid(row=0, column=0, padx=10, pady=10, sticky="nsw")
        self.Label3_2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.Label3_3.grid(row=0, column=2, padx=10, pady=10, sticky="nse")
        self.Label4_1.grid(row=0, column=0, padx=10, pady=10, sticky="nsw")
        self.Label4_2.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        self.Label4_3.grid(row=0, column=2, padx=10, pady=10, sticky="nse")

        # frame1
        self.exit_button = tk.Button(self.frame1, text="X", command=lambda: self.exit_clicked(), width=4)
        self.settings_button = tk.Button(self.frame1, image=S_image, command=self.open_settings_window, width=50)
        # Label2
        self.output_list = tk.Listbox(self.Label2_1)
        self.video_label = tk.Label(self.Label2_2)
        # Label3
        self.tray_text = tk.Label(self.Label3_1, text="ㅤㅤ트레이 버튼ㅤㅤ")
        self.ten_by_ten_button = tk.Button(self.Label3_1, text="10x10", command=lambda: self.capture_button_clicked("10x10"), height=2, width=15)
        self.five_by_ten_button = tk.Button(self.Label3_1, text="5x10", command=lambda: self.capture_button_clicked("5x10"), height=2, width=15)
        self.S_ten_by_ten_button = tk.Button(self.Label3_1, text="S_10x10", command=lambda: self.capture_button_clicked("S_10x10"), height=2, width=15)
        self.two_hundred_button = tk.Button(self.Label3_1, text="200-1", command=lambda: self.capture_button_clicked("200-1"), height=2, width=15)
        self.unit_text = tk.Label(self.Label3_3, text="ㅤㅤ부품 이름 (이미지 저장용)ㅤㅤ")
        self.text_box = tk.Text(self.Label3_3, height=1,width=25)
        self.unit_button = tk.Button(self.Label3_3, text="입력", command=self.print_text, height=2, width=20)
        # Label4
        self.threshold_text = tk.Label(self.Label4_1, text="ㅤㅤ임계값 설정ㅤㅤ")
        self.threshold_slider = tk.Scale(self.Label4_1, from_=0.1, to=400, resolution=0.1, orient=tk.HORIZONTAL, length=690, variable=self.threshold, width=60)
        self.combobox_text = tk.Label(self.Label4_3, text="ㅤㅤ임계값 리스트ㅤㅤ")
        self.combobox = ttk.Combobox(self.Label4_3, values=list(self.threshold_values.keys()), style='TCombobox', width=25)
        # frame5
        self.TW_logo_image = tk.Label(self.frame5)

        # 디자인
        self.frame1.configure(self.back_style)
        self.frame2_1.configure(self.back_style)
        self.frame2_2.configure(self.back_style)
        self.frame3.configure(self.back_style)
        self.frame4.configure(self.back_style)
        self.frame5.configure(self.root_style)
        self.Label3_1.configure(self.back_style)
        self.Label3_2.configure(self.back_style)
        self.Label3_3.configure(self.back_style)
        self.Label4_1.configure(self.back_style)
        self.Label4_2.configure(self.back_style)
        self.Label4_3.configure(self.back_style)
        # frame1
        self.exit_button.configure(self.exit_label_style)
        self.settings_button.configure(self.label_style)
        self.settings_button.image = S_image
        # Lable2
        self.video_label.configure(self.back_style)
        self.output_list.configure(self.label_style)
        self.scrollbar = tk.Scrollbar(self.output_list, orient=tk.VERTICAL)
        self.xscrollbar = tk.Scrollbar(self.output_list, orient=tk.HORIZONTAL)
        # Lable3
        self.tray_text.configure(self.text_label_style)
        self.ten_by_ten_button.configure(self.label_style)
        self.five_by_ten_button.configure(self.label_style)
        self.S_ten_by_ten_button.configure(self.label_style)
        self.two_hundred_button.configure(self.label_style)
        self.unit_text.configure(self.text_label_style)
        self.text_box.configure(self.textbox_style)
        self.unit_button.configure(self.label_style)
        # Lable4
        self.threshold_text.configure(self.text_label_style)
        self.threshold_slider.configure(self.label_style)
        self.combobox_text.configure(self.text_label_style)
        # frame5
        self.TW_logo_image.configure(self.root_style, image=L_image)
        self.TW_logo_image.image = L_image
        
        # 위치
        # frame1
        self.exit_button.pack(side=tk.RIGHT, fill=tk.Y)
        self.settings_button.pack(side=tk.LEFT, fill=tk.Y)
        # Lable2
        self.output_list.pack(fill=tk.BOTH, expand=True)
        self.scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.xscrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.video_label.pack(fill=tk.BOTH, expand=True)
        # Lable3
        self.tray_text.pack(side=tk.LEFT, fill=tk.BOTH)
        self.ten_by_ten_button.pack(side=tk.LEFT, fill=tk.BOTH)
        self.five_by_ten_button.pack(side=tk.LEFT, fill=tk.BOTH)
        self.S_ten_by_ten_button.pack(side=tk.LEFT, fill=tk.BOTH)
        self.two_hundred_button.pack(side=tk.LEFT, fill=tk.BOTH)
        self.unit_text.pack(side=tk.LEFT, fill=tk.BOTH)
        self.text_box.pack(side=tk.LEFT, fill=tk.BOTH)
        self.unit_button.pack(side=tk.LEFT, fill=tk.BOTH)
        # Lable4
        self.threshold_text.pack(side=tk.LEFT, fill=tk.BOTH)
        self.threshold_slider.pack(side=tk.LEFT, fill=tk.BOTH)
        self.combobox_text.pack(side=tk.LEFT, fill=tk.BOTH)
        self.combobox.pack(side=tk.LEFT, fill=tk.BOTH)
        # frame5
        self.TW_logo_image.pack(side=tk.RIGHT, fill=tk.Y)

        # 콤보박스 폰트
        self.combobox['font'] = font.Font(family="Helvetica", size=15)  # 폰트 크기 조정
        # 엔터 키 이벤트에 대한 바인딩을 설정
        self.text_box.bind("<Return>", lambda e: "break")
        self.combobox.bind("<<ComboboxSelected>>", self.on_combobox_select)
        self.combobox.bind("<Button-1>", self.read_file)
        self.root.bind('<KeyPress>', self.on_key_press)
        # Listbox와 Scrollbar 연결
        self.output_list.config(yscrollcommand = self.scrollbar.set)
        self.scrollbar.config(command = self.output_list.yview)
        self.output_list.config(xscrollcommand=self.xscrollbar.set)
        self.xscrollbar.config(command = self.output_list.xview)

        # 기타 변수
        self.button_image = None
        self.end = None
        self.unit_name = "empty"
        self.current_button = ""
        self.read_file(event="")

    def open_settings_window(self):
        S_root = tk.Tk()
        self.settings_window = SettingsWindow(S_root)
        
    def read_file(self, event):
        self.threshold_values = {}
        document_folder = os.path.expanduser("~/Documents")
        file_path = os.path.join(document_folder, "TW", "settings.txt")
        # 파일 열기
        try:
            with open(file_path, "r") as file:
                for line in file:
                    # 각 줄을 ":"를 기준으로 분리
                    parts = line.strip().split(":")
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = float(parts[1].strip())  # 숫자로 변환
                        self.threshold_values[key] = value
            self.combobox['values']=list(self.threshold_values.keys())
        except FileNotFoundError:
            pass

    def print_text(self):
        input_text = str(self.text_box.get("1.0", tk.END)).strip()
        if input_text:
            if SettingsWindow.is_valid_input(input_text):
                try:
                    self.unit_name = str(self.text_box.get("1.0", tk.END)).strip()
                    self.output_list.insert(tk.END, f"제품 : {self.unit_name} 입력 완료ㅤㅤ")
                    self.IS = save(self.unit_name, program = "Tray", function = "Origin", current_button=self.current_button)
                except:
                    self.output_list.insert(tk.END, f"에러 발생: 제품 입력을 재시도 해주세요ㅤㅤ")
            else:
                self.output_list.insert(tk.END, "ㅤㅤ")
                self.output_list.insert(tk.END, "에러 발생: 제품 입력에 지원하지 않는 문자가 포함되어 있습니다.ㅤㅤ")
                self.output_list.insert(tk.END, "영어 또는 숫자 일부 기호만 사용.ㅤㅤ")
                self.output_list.insert(tk.END, "ㅤㅤ")
                self.output_list.insert(tk.END, "ㅤ! @ # $ % ^ _ - . 만을 사용하여 제품 이름을 입력해 주십시오.ㅤㅤ")
        else:
            self.output_list.insert(tk.END, f"제품 입력 안됨, 텍스트 박스에 제품을 입력해 주세요ㅤㅤ")

    # 콤보박스 선택 시 실행되는 함수
    def on_combobox_select(self, event):
        selected_value = self.combobox.get()
        self.threshold.set(self.threshold_values[selected_value])

    def exit_clicked(self):
        result = messagebox.askquestion("Exit Confirmation", "프로그램을 종료 하시겠습니까?")
        if result == "yes":
            self.end = 0
            self.root.destroy()
            sys.exit(0)

    def on_key_press(self, event):
        if event.keysym == 'Left':
            self.threshold_slider.set(self.threshold_slider.get() - 0.1)
        elif event.keysym == 'Right':
            self.threshold_slider.set(self.threshold_slider.get() + 0.1)
    
    def isave(self, function: str, image: np.ndarray):
        self.IS = save(self.unit_name, program = "Tray", function = function, current_button = self.current_button)
        self.photo_path = self.IS.tray_image_save(image)
        self.output_list.insert(tk.END, f"{self.photo_path} 경로에 이미지 저장 완료ㅤㅤ")
        self.output_list.insert(tk.END, "ㅤㅤ")

    def capture_button_clicked(self, button_name: str):
        self.current_button = button_name
        IPE = IP.Edit(frame, self.current_button, self.threshold.get())
        image = IPE.edit_image()
        self.isave("Origin", image)

        IPpod = IP.perform_object_detection(image, self.current_button, self.threshold.get())
        result_image = IPpod.cell_check()
        result_image = IC.Image_empty(result_image)
        result_image = IPpod.num_text(result_image)
        self.isave("Scan", result_image)

        plt.rcParams['figure.dpi'] = 100
        monitors = get_monitors()
        screen_width, screen_height = monitors[0].width, monitors[0].height
        fig = plt.figure(figsize=(screen_width/100, screen_height/100))

        scan_image = cv2.imread(self.photo_path)
        scan_image = cv2.cvtColor(scan_image, cv2.COLOR_BGR2RGB)

        # 두 번째 이미지를 오른쪽에 표시
        plt.subplot(1,1, 1)
        plt.imshow(scan_image)
        plt.axis('off')  # 선택적으로 축을 표시하지 않도록 설정할 수 있습니다.
        plt.title("Scan")
        fig_manager = plt.get_current_fig_manager()
        fig_manager.window.state('zoomed')

        plt.show()

    def video_label_update(self, image: np.ndarray):
        self.image = image
        self.video_label.configure(image=image)
        self.video_label.image = image
        self.photo_path = ""


if __name__ == "__main__":
    T = TW()
    if T() == True:
        subapp = QApplication(sys.argv)
        loading_screen = LoadingScreen()
        loading_screen.show()

        root = tk.Tk()
        app = MainView(root)
        cam = Camera()
        cam.open_camera()  # 카메라 열기
        cam.set_cap_size(app.screen_width, (app.screen_width*9)//16)
        app.output_list.insert(tk.END, f"연결된 카메라 리스트")
        app.output_list.insert(tk.END, f"{cam.cameras_list()}ㅤㅤ")
        app.output_list.insert(tk.END, f"ㅤ")
        app.output_list.insert(tk.END, f"{cam.name} 카메라 활성화")
        app.output_list.insert(tk.END, f"카메라 해상도 {app.screen_width}, {(app.screen_width*9)//16}")
        app.output_list.insert(tk.END, f"ㅤ")
        loading_screen.close()

        IC = ImageCV()

        while cam.is_camera_open():
            frame = cam.get_frame()  # 프레임 가져오기
            copy_image = copy.deepcopy(frame)
            app.button_image = copy_image

            # 비디오 레이블 이미지
            height, width = frame.shape[:2]
            cut_width = width//9
            frame = frame[:,cut_width : width-cut_width]
            fream_Resolution = IC.Scale_Resolution(frame, 0.7)
            video_label_image = cv2.resize(frame, fream_Resolution)

            try:
                image_tk = ImageTk.PhotoImage(
                                Image.fromarray(
                                    cv2.cvtColor(video_label_image, cv2.COLOR_BGR2RGB)
                                )) # PIL 이미지를 Tkinter PhotoImage 객체로 변환

                # 레이블에 새 이미지 업데이트
                app.video_label_update(image_tk)
                root.update()
            except tk.TclError:
                pass
        
        # 카메라 종료
        cv2.destroyAllWindows()
        cam.release_camera()
        root.mainloop()
        sys.exit(subapp.exec_())

    elif T() == False:
        messagebox.showinfo("SM ERROR", "해당 프로그램은 설정된 컴퓨터에서 실행 가능합니다.\n변경을 원할 경우 업체에 요청하시길 바랍니다.")

    elif T() == 2:
        messagebox.showinfo("OS ERROR", "해당 프로그램은 Windows10 이상에서만 실행 가능합니다.")

    else:
        messagebox.showinfo("ERROR", T())