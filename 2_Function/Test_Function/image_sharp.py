import cv2
import numpy as np
from PIL import Image, ImageTk

def filter_black_white(image):
    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 이미지 이진화
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 이진화된 이미지를 BGR 형식으로 변환
    binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

    # 원본 이미지와 이진화된 이미지를 비트 AND 연산
    result = cv2.bitwise_and(image, binary_bgr)

    return result

def sharpen_image(image, amount, radius):
    blurred = cv2.GaussianBlur(image, (0, 0), radius)
    sharpened = cv2.addWeighted(image, 1.0 + amount, blurred, -amount, 0)
    return sharpened

# 이미지 불러오기
image = cv2.imread('./160728_Origin.jpg')

# 이미지 선명화 적용
B_image = filter_black_white(image)
sharpened_image = sharpen_image(image, 2.0, 2.0)
sharpened_B_image = filter_black_white(sharpened_image)

# 선명화된 이미지 출력
cv2.imshow('Sharpened Image+black', sharpened_B_image)
cv2.imshow('Sharpened Image', sharpened_image)
cv2.imshow('B_orign Image', B_image)
cv2.imshow('orign Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()