import sys
import cv2
import time
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from PyQt5.QtWidgets import QApplication

from IMG.IPP import ImageCV
from IMG.camera import Camera

from TW.TWSM import TW
from TW.UI import MainView
from TW.Loading import LoadingScreen

class Main:
    def __init__(self) -> None:
        """
        TW, Camera, ImageCV 인스턴스들을 초기화합니다.
        """
        self.T = TW()
        self.cam = Camera()
        self.IC = ImageCV()

    def check(self) -> bool:
        """
        TWSM를 이용하여 프로그램이 실행 가능한 환경인지를 확인합니다.
        실행 가능한 환경인 경우 True를 반환하고 조건에 맞지않을 경우 메시지박스를 출력하여 실행 불가능한 이유를 보여줍니다.
        """
        if self.T() == True:
            return True
        elif self.T() == False:
            messagebox.showinfo("SM ERROR", "해당 프로그램은 설정된 컴퓨터에서 실행 가능합니다. 변경을 원할 경우 업체에 요청하시길 바랍니다.")
        elif self.T() == 2:
            messagebox.showinfo("OS ERROR", "해당 프로그램은 Windows10 이상에서만 실행 가능합니다.")
        else:
            messagebox.showinfo("ERROR", self.T())
    
    def start(self, loading_screen: LoadingScreen) -> None:
        """
        애플리케이션을 시작하는 함수입니다. 이 함수는 tkinter와 QApplication 객체를 생성하고,
        카메라를 설정하고 연결합니다. 그리고 카메라에서 프레임을 가져와서 처리하는 작업을 반복합니다.
        마지막으로, 모든 작업이 끝나면 카메라를 종료하고 자원을 해제합니다.
        """
        root = tk.Tk()
        root.withdraw()
        app = MainView(root)
        subapp = QApplication(sys.argv)
        self.cam.open_camera()
        self.cam.set_cap_size(app.screen_width, (app.screen_width*9)//16)
        time.sleep(1)
        loading_screen.close()
        app.output_list.insert(tk.END, f"연결된 카메라 리스트")
        app.output_list.insert(tk.END, f"{self.cam.cameras_list()}ㅤㅤ")
        app.output_list.insert(tk.END, f"ㅤ")
        app.output_list.insert(tk.END, f"{self.cam.name} 카메라 활성화")
        app.output_list.insert(tk.END, f"카메라 해상도 {app.screen_width}, {(app.screen_width*9)//16}")
        app.output_list.insert(tk.END, f"ㅤ")
        while self.cam.is_camera_open() and app.end:
            frame = self.cam.get_frame()
            app.frame = frame
            app.button_image = frame
            frame_Resolution = self.IC.Scale_Resolution(frame, 0.6)
            video_label_image = cv2.resize(frame, frame_Resolution)
            try:
                image_tk = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(video_label_image, cv2.COLOR_BGR2RGB)))
                app.video_label_update(image_tk)
                root.update()
            except:
                pass
        cv2.destroyAllWindows()
        self.cam.release_camera()
        root.mainloop()
        sys.exit()

if __name__ == "__main__":
    """
    LoadingScreen 클래스를 실행 시킨 후
    Main 클래스의 인스턴스를 생성하고
    실행 가능한 환경인지 확인합니다.
    실행 가능 환경이라면 start 메소드를 호출하여 프로그램을 시작합니다.
    """
    loading_screen = LoadingScreen()
    loading_screen.show()
    M = Main()
    HW_result = M.check()
    if HW_result == True:
        M.start(loading_screen)
