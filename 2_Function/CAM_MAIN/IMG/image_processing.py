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

class Edit:
    def __init__(self, frame, current_button, threshold) -> None:
        self.frame = frame
        self.current_button = current_button
        self.threshold = threshold
        self.root = None
        self.part_cell_num = 0
        self.empty_cell_num = 0
        self.m = 5

    def get_monitor_resolution(self):
        app = QApplication([])
        screen = app.primaryScreen()
        screen_width = screen.size().width()
        screen_height = screen.size().height()
        return screen_width, screen_height

    def remapping_image(self, frame: np.ndarray) -> np.ndarray:
        # 왜곡 계수 설정
        k1, k2, k3 = -0.02, 0.0, 0.0  # 핀큐션 왜곡
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

    def on_slider_blurred(self, frame: np.ndarray) -> np.ndarray:
        blur_value = self.m
        # 블러 적용
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)

        _, binary = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    # 이미지 윤곽선 처리
    def preprocessing_image(self, image: np.ndarray) -> np.ndarray:
        # 그레이스케일 변환
        binary = self.on_slider_blurred(image)
        # 윤곽선 찾기
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = []

        # 윤곽선을 순회하며 면적이 일정 크기 이상인 윤곽선만 저장
        min_contour_area = 100000
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_contour_area:
                filtered_contours.append(self.coordinate_extraction(contour))
        return filtered_contours

    # 꼭지점 좌표 추출
    def coordinate_extraction(self, contour: np.ndarray) -> np.ndarray:
        # 윤곽선 근사화
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx
    
    def adjust_location(self, location, x_factor, y_factor, x_1_factor, y_1_factor):
        x = int(location[0][0] * x_factor)
        x_1 = int(location[0][0] * x_1_factor)
        y = int(location[0][1] * y_factor)
        y_1 = int(location[0][1] * y_1_factor)

        location[0][0] += x_1
        location[2][0] -= x
        location[0][1] += y
        location[2][1] += y
        location[1][0] += x_1
        location[3][0] -= x
        location[1][1] -= y_1
        location[3][1] -= y_1
        return location

        
    # perform_object_detection()에서 이미지 전처리 및 탐색 전 셀 구분선 생성
    def edit_image(self) -> np.ndarray:
        image = self.frame
        if image is None:
            sys.exit()
        image = self.remapping_image(image)
        dst = image
        height, width = image.shape[:2]

        # 꼭지점 좌표 추출용 배열
        location_nparr = []
        approx = []

        while len(approx) != 4:
            approx = self.preprocessing_image(image)
            approx = approx[0]
            if len(approx) != 4:
                approx = [[]]
                self.m += 2
                if self.m == 101:
                    break
                else:
                    continue

            else:
                try:
                    approx = np.array(sorted(approx, key=lambda x: x[0][0] + x[0][1]))
                    for point in approx:
                        x, y = point[0]
                        location_nparr.append([x, y+2.0])
                    location = np.array(location_nparr, np.float32)
                    
                    if self.current_button in ["200-1", "S_10x10"]:
                        location = self.adjust_location(location, 0.17, 35, 0.17, 35)
                    elif self.current_button in ["10x10", "5x10"]:
                        location = self.adjust_location(location, 0.07, 0, 0.07, 0)

                    # location2 = np.array([[0, 0], [0, height], [width, 0], [width, height]], np.float32)
                    # pers = cv2.getPerspectiveTransform(location, location2)
                    # dst = cv2.warpPerspective(image, pers, (width, height))
                    x, y, w, h = cv2.boundingRect(np.array(location))
                    dst = image[y:y+h, x:x+w]
                    return dst, ""
                except Exception as e:
                    return dst, f"Error : {e}"
        return dst, "None"

class perform_object_detection:
    def __init__(self, frame: np.ndarray, current_button: str, threshold: float) -> None:
        self.current_button = current_button
        self.frame = frame
        self.threshold = threshold
        self.part_cell_num = 0
        self.empty_cell_num = 0
        self.height, self.width = frame.shape[:2]
        self.cell_width = 0
        self.cell_height = 0
        self.empty_cells = []
        self.part_cells = []
        self.contour_cell = []

    def tray_cell(self) -> None:
        self.cell_width = self.width // 10
        if self.current_button == "10x10":
            self.cell_height = self.height // 10

        elif self.current_button == "5x10":
            self.cell_height = self.height // 5

        elif self.current_button == "S_10x10":
            self.cell_height = self.height // 10

        elif self.current_button == "200-1":
            self.cell_height = int((self.height / 13))
            self.cell_width = int((self.width / 16))

    def cell_i(self, i: int) -> float:# cell 좌표 계산 함수
        cell_x = i * self.cell_width
        cell_y = i * self.cell_height
        return cell_x, cell_y

    def cell_draw(self, start_x: float, start_y: float, end_x: float, end_y: float) -> None:
        cv2.line(self.frame,
                    (start_x, start_y),
                    (end_x, end_y),
                    (205, 0, 205), 1)
        
    def line_draw(self) -> None:
        if self.current_button == "200-1":
            # 구분선 가로와 세로
            for i in range(1, 16):
                cell_x, cell_y = self.cell_i(i)
                if  i <= 16:
                    self.cell_draw(cell_x, 0, cell_x, self.height)
                    
                if i <= 12:
                    self.cell_draw(0, cell_y, self.width, cell_y)
                
        else:
            for i in range(1, self.width // self.cell_width):
                cell_x = i * self.cell_width
                self.cell_draw(cell_x, 0, cell_x, self.height)
                if self.cell_height <= i * self.cell_height:
                    cell_y = i * self.cell_height
                    self.cell_draw(0, cell_y, self.width, cell_y)

    def image_inRange(self) -> None:
        if self.current_button == "10x10" or "5x10":
            C1_frame = copy.deepcopy(self.frame)
            # 이미지의 색상 경계 설정
            lower_color = np.array([0], dtype=np.uint8)     # 검은색 (빈 칸)
            upper_color = np.array([70], dtype=np.uint8)    # 어두운 색상 (부품이 있는 칸)

            brightness_factor = 1  # 밝기를 증가시킬 비율 (1.0보다 크면 밝아지고, 1.0보다 작으면 어두워집니다.)
            brightened_frame = cv2.convertScaleAbs(C1_frame, alpha=brightness_factor, beta=0)

            self.mask = cv2.inRange(brightened_frame, lower_color, upper_color)

        elif self.current_button == "200-1" or "S_10x10":
            C2_frame = copy.deepcopy(self.frame)
            # Noise removal
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            bin_img = cv2.morphologyEx(C2_frame, cv2.MORPH_OPEN, kernel)

            # Image grayscale conversion and contrast adjustment
            bin_img = np.clip((1 + 2.3) * bin_img - 128 * 2.3, 0, 255).astype(np.uint8)

            gray_img = cv2.cvtColor(bin_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
            ret, binary_img = cv2.threshold(gray_img,  245 , 255, cv2.THRESH_BINARY_INV)
            inverted_binary_img = cv2.bitwise_not(binary_img)
            self.mask = cv2.inRange(inverted_binary_img, np.array([0], dtype=np.uint8), np.array([1], dtype=np.uint8))

    def unit_contour(self, i: int, j: int) -> None:
        cell_roi, cell_contours, contour_area = None, None, None
        cell_y = i * self.cell_height
        cell_x = j * self.cell_width
    
        cell_roi = self.mask[cell_y : cell_y + self.cell_height, cell_x:cell_x + self.cell_width]
        cell_contours, _ = cv2.findContours(cell_roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 윤곽선 면적 계산
        contour_area = sum(cv2.contourArea(contour) for contour in cell_contours)

        if sum(cv2.contourArea(contour) for contour in cell_contours) > 1:
            # 윤곽선 면적 계산
            contour_area = sum(cv2.contourArea(contour) for contour in cell_contours)
            # 부품이 두 개 이상 들어간 칸인지 확인
            num_contours = len(cell_contours)
            self.contour_cell.append(num_contours)

            # print(contour_area) # 유닛 면적 확인용
            # 면적이 임계값을 곱한 값의 이상인지 확인하는 조건문
            if contour_area > self.cell_width * self.cell_height * (self.threshold * 0.001):
                    self.part_cells.append((j, i))
                    self.part_cell_num+=1
        else:
            self.empty_cells.append((j, i))
            self.empty_cell_num+=1

    def cell_check(self) -> np.ndarray:
        self.tray_cell()
        self.line_draw()
        self.image_inRange()
        # 윤곽선을 이용하여 빈 칸과 부품이 있는 칸 검출
        if self.current_button == "200-1":
            for i in range(0, 13):
                for j in range(0, 16):
                    if i == 0 and j not in range(4, 16):
                        continue
                    if i == 12 and j not in range(0, 12):
                        continue
                    self.unit_contour(i, j)
        elif self.current_button == "5x10":
            for i in range(5):
                for j in range(10):
                    self.unit_contour(i, j)
        else:
            for i in range(10):
                for j in range(10):
                    self.unit_contour(i, j)
        # 결과 이미지를 생성하여 표시
        result_image = copy.deepcopy(self.frame)
        for cell in self.empty_cells:
            cell_x = cell[0] * self.cell_width
            cell_y = cell[1] * self.cell_height
            # 빨간색 사각형 그리기
            cv2.rectangle(result_image,
                            (cell_x + 15, cell_y + 10),
                            (cell_x + self.cell_width - 20,
                            cell_y + self.cell_height - 10),
                            (0, 0, 255), 2)

        # 부품이 있는 칸에 파란색 원 그리기
        for cell in self.part_cells:
            cell_x = cell[0] * self.cell_width + self.cell_width // 2
            cell_y = cell[1] * self.cell_height + self.cell_height // 2
            '''
            radius = min(cell_width, cell_height) // 3  # 크기를 더 키워줍니다.
            예를 들어 "// 3" 대신 "// 2"로 변경하여 더 큰 원을 그릴 수 있습니다.
            '''
            cv2.circle(result_image, (cell_x, cell_y),
                            radius=self.cell_width//4,
                            color=(255, 0, 0),
                            thickness=2)  # 파란색 원 그리기
        return result_image
    
    def num_text(self, image: np.ndarray) -> np.ndarray:
        empty_cell_text = f"Empty Cells: {self.empty_cell_num}"
        part_cell_text = f"Part Cells: {self.part_cell_num}"

        # Copy the input image to avoid modifying the original
        output_image = np.copy(image)

        # Define the font and position for the text
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.4
        font_color = (255, 255, 255)  # White color
        font_thickness = 1
        x = 30  # X-coordinate of the text
        y = 45  # Y-coordinate of the text

        # Put the text on the image
        output_image = cv2.putText(output_image, empty_cell_text, (x, y), font, font_scale, font_color, font_thickness)
        y += 20  # Move down for the next line of text
        output_image = cv2.putText(output_image, part_cell_text, (x, y+35), font, font_scale, font_color, font_thickness)

        return output_image
