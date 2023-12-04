# Image Pre-Processing
import os
import sys
import cv2
import numpy as np
import copy
from datetime import datetime
from PyQt5.QtWidgets import *

class ImageCV:
    def __init__(self) -> None:
        """
        ImageCV 클래스의 초기화 함수입니다. 밝기, 커널 등을 초기 설정합니다.
        """
        self.brigtness = -30 # 양수 밝게, 음수 어둡게
        self.alp = 1.0
        self.kernel_3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        self.kernel_5 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        self.kernel_ones = np.ones((3,3),np.uint8)
        self.threshold = 0

    def gray(self, image: np.ndarray) -> np.ndarray:
        """
        이미지를 그레이스케일로 변환하는 함수입니다.
        """
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image
        return gray_image

    def BGR(self, image: np.ndarray) -> np.ndarray:
        """
        이미지를 그레이스케일로 변환하는 함수입니다.
        """
        if len(image.shape) == 1:
            BGR_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        else:
            BGR_image = image
        return BGR_image

    def Image_Crop(self, image, pos: np.ndarray, dsize: tuple) -> np.ndarray:
        """
        주어진 위치와 크기에 따라 이미지를 자르는 함수입니다.
        """
        dst = cv2.warpPerspective(image, pos, dsize)
        return cv2.flip(dst, 0)

    def Scale_Resolution(self, image: np.ndarray, Scale: float) -> tuple:
        """
        주어진 배율에 따라 이미지의 해상도를 조정하는 함수입니다.
        """
        height, width = image.shape[:2]
        return (int(width*Scale), int(height*Scale))

    def Image_Slice(self, image: np.ndarray, height_value: float, width_value: float) -> np.ndarray:
        """
        주어진 높이와 너비 값에 따라 이미지를 슬라이스하는 함수입니다.
        """
        height, width = image.shape[:2]
        fix_value = [0,0]
        values = [height_value, width_value]
        for i in range(len(values)):
            if values[i] <= 0.01 and values[i] > 0:
                fix_value[i] = 0
            elif values[i] > 0.01:
                fix_value[i] = 0.01
            elif values[i] <= 0:
                values[i] = 0
                fix_value[i] = 0
        return image[int(height*(values[0]+fix_value[0])):int(height*(1-values[0]+fix_value[0])):
                    ,int(width*values[1]):int(width*(1-values[1]+fix_value[1]))]

    def Brightness(self, image: np.ndarray) -> np.ndarray:
        """
        이미지의 밝기를 조절하는 함수입니다.
        """
        image = (np.int16(image) + self.brigtness).astype(np.uint8)
        return image

    def Dilate(self, image: np.ndarray) -> np.ndarray:
        """
        이미지에 팽창 연산을 적용하는 함수입니다.
        """
        image = np.clip((1.0+self.alp) * image - 128 * self.alp, 0, 255)
        image = cv2.dilate(image,self.kernel_5, iterations = 1)
        return image

    def Pos_by_Img(self, image, pos):
        """
        이미지를 주어진 위치에 맞게 잘라내고, 밝기를 조절하고, 해상도를 조정하는 함수입니다.
        """
        image = self.Image_Crop(image, pos, (900,1200))
        image = self.Image_Slice(image, height_value=0.02, width_value=0.02)
        image = self.Brightness(image)
        result_image_Resolution = self.Scale_Resolution(image, 0.5)
        image = cv2.resize(image, result_image_Resolution)
        return image

    def Histogram_Equalization(self, image: np.ndarray) -> np.ndarray:
        """
        이미지에 히스토그램 평활화를 적용하는 함수입니다.
        """
        gray_image = self.gray(image)
        equalized_image = cv2.equalizeHist(gray_image)
        return equalized_image

    def Contrast_Adjustment(self, image: np.ndarray) -> np.ndarray:
        """
        이미지의 대비를 조절하는 함수입니다.
        """
        alpha = 1.5
        dst = cv2.convertScaleAbs(image, alpha=alpha, beta=(1 - alpha) * 128)
        return dst
    
    def highlight_contours(self, image: np.ndarray, contours: np.ndarray) -> np.ndarray:
        """
        이미지에서 윤곽선을 강조하는 함수입니다.
        """
        mask = np.zeros_like(image)
        cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
        inverse_mask = 255 - mask
        image[inverse_mask == 255] = 255
        return image

    def color_invert(self, image: np.ndarray) -> np.ndarray:
        """
        이미지의 색상을 반전시키는 함수입니다.
        """
        inverted_image = 255 - image
        return inverted_image

    def threshold_brightness(self, image: np.ndarray, threshold: int) -> np.ndarray:
        """
        주어진 임계값 이하의 밝기를 가지는 픽셀을 검은색으로 변경하는 함수입니다.
        """
        gray_image = self.gray(image)
        # Apply the threshold to set pixels below the threshold to black
        _, thresholded_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        return thresholded_image

    def White_Mask(self, image: np.ndarray, thresh: np.ndarray) -> np.ndarray:
        """
        이미지에 흰색 마스크를 적용하는 함수입니다.
        """
        mask = np.zeros_like(image)
        mask[thresh  == 255] = 255
        white_parts = cv2.bitwise_and(image, mask)
        return white_parts

    def Background_Area(self, image: np.ndarray) -> np.ndarray:
        """
        이미지에서 배경 영역을 검출하는 함수입니다.
        """
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        image = self.gray(image)
        dist = cv2.distanceTransform(image, cv2.DIST_L2, 5)
        # foreground area
        _, sure_fg = cv2.threshold(dist, 0.46* dist.max(), 255, cv2.THRESH_BINARY)
        sure_bg = cv2.dilate(sure_fg, kernel, iterations=1)
        sure_bg = sure_bg.astype(np.uint8)
        return sure_bg

    def Image_empty(self, image: np.ndarray) -> np.ndarray:
        """
        이미지의 상단에 빈 공간을 추가하는 함수입니다.
        """
        height, width, channels = image.shape
        # 상단에 추가할 빈 공간의 높이 설정
        top_padding =120  # 원하는 높이로 설정
        # 여백을 추가할 새로운 이미지 생성
        new_height = height + top_padding
        new_image = np.zeros((new_height, width, channels), dtype=np.uint8)
        # 기존 이미지를 새로운 이미지의 아래로 복사
        new_image[top_padding:, :] = image
        return new_image

class save:
    def __init__(self, unit_name: str, program: str, function: str, current_button: str) -> None:
        """
        save 클래스의 초기화 함수입니다.
        유닛 이름, 프로그램, 기능, 현재 버튼 등을 초기 설정합니다.
        """
        self.program = program
        self.function = function
        self.unit = unit_name
        self.current_button = current_button
        # Documents(문서) 폴더 경로
        self.documents_folder = os.path.join(os.path.expanduser("~"))
        self.main_folder = os.path.join(self.documents_folder, "TW")
        self.sub_folder = os.path.join(self.main_folder, self.program)
        if self.program == "Tray":
            self.button_folder = os.path.join(self.sub_folder, self.current_button)
            self.date_folder = os.path.join(self.button_folder, datetime.today().strftime('%y%m%d'))
        else:
            self.date_folder = os.path.join(self.sub_folder, datetime.today().strftime('%y%m%d'))
        self.unit_folder = os.path.join(self.date_folder, self.unit)

    def micro_image_save(self, frame: np.ndarray) -> str:
        """
        마이크로 이미지를 저장하는 함수입니다.
        이미지는 유닛 이름, 프로그램, 기능, 현재 버튼 등에 따라 적절한 폴더에 저장됩니다.
        """
        for f in [self.main_folder, self.sub_folder, self.date_folder, self.unit_folder]:
            if not os.path.exists(f):
                os.mkdir(f)
        # 이미지를 읽어옴
        filename = f"{datetime.today().strftime('%H%M%S')}_{self.function}.jpg"
        photo_path = os.path.join(self.unit_folder, filename)
        # 이미지를 저장
        cv2.imwrite(photo_path, frame)
        return photo_path

    def tray_image_save(self, frame: np.ndarray) -> str:
        """
        트레이 이미지를 저장하는 함수입니다.
        이미지는 유닛 이름, 프로그램, 기능, 현재 버튼 등에 따라 적절한 폴더에 저장됩니다.
        """
        # 폴더가 이미 존재하는지 확인 후 생성
        for f in [self.main_folder, self.sub_folder, self.button_folder, self.date_folder, self.unit_folder]:
            if not os.path.exists(f):
                os.mkdir(f)
        # 이미지를 읽어옴
        filename = f"{datetime.today().strftime('%H%M%S')}_{self.function}.jpg"
        photo_path = os.path.join(self.unit_folder, filename)
        # 이미지를 저장
        cv2.imwrite(photo_path, frame)
        return photo_path

class Edit:
    def __init__(self, frame: np.ndarray, current_button: str, threshold: float) -> None:
        """
        Edit 클래스의 초기화 함수입니다.
        프레임, 버튼 상태, 임계값 등을 초기 설정합니다.
        """
        self.frame = frame
        self.current_button = current_button
        self.threshold = threshold
        self.root = None
        self.part_cell_num = 0
        self.empty_cell_num = 0
        self.m = 5

    def get_monitor_resolution(self):
        """
        모니터의 해상도를 반환하는 함수입니다.
        """
        app = QApplication([])
        screen = app.primaryScreen()
        screen_width = screen.size().width()
        screen_height = screen.size().height()
        return screen_width, screen_height

    def remapping_image(self, frame: np.ndarray) -> np.ndarray:
        """
        이미지를 리매핑하는 함수입니다.
        이미지의 왜곡을 제거합니다.
        """
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
        """
        이미지에 가우시안 블러를 적용하는 함수입니다.
        슬라이더의 값을 기반으로 블러를 적용합니다.
        """
        blur_value = self.m
        # 블러 적용
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred_frame = cv2.GaussianBlur(gray, (blur_value, blur_value), 0)
        _, binary = cv2.threshold(blurred_frame, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary

    def preprocessing_image(self, image: np.ndarray) -> np.ndarray:
        """
        이미지를 전처리하는 함수입니다.
        그레이스케일 변환과 윤곽선 검출을 수행합니다.
        """
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

    def coordinate_extraction(self, contour: np.ndarray) -> np.ndarray:
        """
        주어진 윤곽선에서 꼭지점 좌표를 추출하는 함수입니다.
        """
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        return approx

    def adjust_location(self, location, x_factor, y_factor, x_1_factor, y_1_factor):
        """
        주어진 위치를 조정하는 함수입니다. 위치 인수와 팩터 인수를 이용하여 위치를 조정합니다.
        """
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

    def edit_image(self) -> np.ndarray:
        """
        이미지를 편집하는 함수입니다.
        왜곡 제거, 윤곽선 검출, 위치 조정 등의 작업을 수행합니다.
        """
        image = self.frame
        if image is None:
            sys.exit()
        image = self.remapping_image(image)
        dst = image
        # 꼭지점 좌표 추출용 배열
        location_nparr = []
        approx = []
        while len(approx) != 4:
            approx = self.preprocessing_image(image)
            try:
                approx = approx[0]
            except:
                pass
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
        """
        perform_object_detection 클래스의 초기화 함수입니다.
        프레임, 버튼 상태, 임계값 등을 초기 설정합니다.
        """
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
        """
        트레이 내의 셀 크기를 계산하는 함수입니다.
        현재 버튼의 상태에 따라 셀의 크기가 결정됩니다.
        """
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

    def cell_i(self, i: int) -> float:
        """
        셀의 좌표를 계산하는 함수입니다.
        주어진 인덱스에 대해 셀의 x, y 좌표를 계산하여 반환합니다.
        """
        cell_x = i * self.cell_width
        cell_y = i * self.cell_height
        return cell_x, cell_y

    def cell_draw(self, start_x: float, start_y: float, end_x: float, end_y: float) -> None:
        """
        주어진 좌표에 따라 셀에 선을 그리는 함수입니다.
        """
        cv2.line(self.frame,
                    (start_x, start_y),
                    (end_x, end_y),
                    (205, 0, 205), 1)

    def line_draw(self) -> None:
        """
        셀에 선을 그리는 함수입니다.
        현재 버튼의 상태에 따라 적절한 방식으로 선을 그립니다.
        """
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
        """
        이미지의 색상 범위를 설정하는 함수입니다.
        현재 버튼의 상태에 따라 적절한 방식으로 범위를 설정합니다.
        """
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
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            bin_img = cv2.morphologyEx(C2_frame, cv2.MORPH_OPEN, kernel)
            bin_img = np.clip((1 + 2.3) * bin_img - 128 * 2.3, 0, 255).astype(np.uint8)
            gray_img = cv2.cvtColor(bin_img, cv2.COLOR_BGR2GRAY)
            ret, binary_img = cv2.threshold(gray_img,  245 , 255, cv2.THRESH_BINARY_INV)
            inverted_binary_img = cv2.bitwise_not(binary_img)
            self.mask = cv2.inRange(inverted_binary_img, np.array([0], dtype=np.uint8), np.array([1], dtype=np.uint8))

    def unit_contour(self, i: int, j: int) -> None:
        """
        주어진 인덱스에 대해 윤곽선을 찾고, 빈 칸과 부품이 있는 칸을 분류하는 함수입니다.
        """
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
        """
        셀을 확인하고 결과 이미지를 생성하는 함수입니다.
        빈 칸과 부품이 있는 칸에 대해 적절한 그래픽 요소를 그립니다.
        """
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
        """
        이미지 위에 텍스트를 그리는 함수입니다.
        빈 칸과 부품이 있는 칸의 수를 이미지 위에 표시합니다.
        """
        empty_cell_text = f"Empty Cells: {self.empty_cell_num}"
        part_cell_text = f"Part Cells: {self.part_cell_num}"
        output_image = np.copy(image)
        font = cv2.FONT_HERSHEY_DUPLEX
        font_scale = 1.4
        font_color = (255, 255, 255)
        font_thickness = 1
        x = 30
        y = 45
        output_image = cv2.putText(output_image, empty_cell_text, (x, y), font, font_scale, font_color, font_thickness)
        y += 20
        output_image = cv2.putText(output_image, part_cell_text, (x, y+35), font, font_scale, font_color, font_thickness)
        return output_image


'''
-------------------------------------------테스트-------------------------------------------
'''

# if __name__ == "__main__":
#     image = cv2.imread("C:/Users/sjmbe/TW/TEST/230908/161906_Micro.jpg")
#     IC = ImageCV()
#     cv2.imshow("test",IC.Binarization(IC.Brightness(image)))
#     cv2.waitKey(0)
