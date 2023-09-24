import cv2
import numpy as np

# 이미지 불러오기
image = cv2.imread('D:/test_10x10_image.png')

# 이미지 크기 및 중심 좌표 계산
height, width, _ = image.shape
center_x = int(width / 2)
center_y = int(height / 2)

# 영역 구분선 그리기
image_with_lines = image.copy()
cv2.line(image_with_lines, (center_x, 0), (center_x, height), (0, 0, 255), 2)
cv2.line(image_with_lines, (0, center_y), (width, center_y), (0, 0, 255), 2)

# 영역별로 이미지 쪼개기
top_left = image[:center_y, :center_x]
top_right = image[:center_y, center_x:]
bottom_left = image[center_y:, :center_x]
bottom_right = image[center_y:, center_x:]

# 윤곽선 찾기
gray = cv2.cvtColor(top_left, cv2.COLOR_BGR2GRAY)
_, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 최외곽 윤곽선 꼭지점 찾기
external_contour = min(contours, key=cv2.contourArea)
x, y, _, _ = cv2.boundingRect(external_contour)
top_left_corner = (x, y)

# top_left_corner 좌표를 image_with_lines 이미지에 점으로 표시
cv2.circle(image_with_lines, top_left_corner, 5, (0, 255, 0), -1)


# 결과 출력
cv2.imshow('Image with Lines', image_with_lines)
cv2.waitKey(0)
cv2.destroyAllWindows()
