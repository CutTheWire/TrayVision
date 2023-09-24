import cv2
import numpy as np

# 윤곽선 사각형으로 다듬는 함수2_기능/CAM 기능 버전/CAM_4.py
def approximate_contour(contour: np.ndarray, epsilon_ratio: float = 0.04) -> np.ndarray:
    epsilon = epsilon_ratio * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    return approx

# 이미지 불러오기
image = cv2.imread('./5x10_Origin_Image.jpg')

# 이미지 크기 조정
image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)

# 이미지 전처리 (그레이스케일 변환)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
kernel_size = 5

brightness_factor = 2  # 밝기를 증가시킬 비율 (1.0보다 크면 밝아지고, 1.0보다 작으면 어두워집니다.)
brightened_frame = cv2.convertScaleAbs(gray.copy(), alpha=brightness_factor, beta=0)
blurred_frame = cv2.GaussianBlur(brightened_frame, (kernel_size, kernel_size), 0)

# 경계선 감지 (Canny edge detection)
edges = cv2.Canny(gray, 10, 83)
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

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
if list(area_dictionary.values())[0] is not None:
    main_area = list(area_dictionary.values())[0]

approx_image = approximate_contour(main_area)

# 사각형 그리기
if len(approx_image) == 4:  # 꼭지점이 4개일 때만 사각형으로 그리기
    cv2.polylines(image, [approx_image], True, (0, 0, 255), 2)  # 빨간색으로 윤곽선 그리기

# 결과 출력
cv2.imshow('Contours', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
