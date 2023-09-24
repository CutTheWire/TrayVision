import cv2
import numpy as np

def fill_black_area(image, threshold):
    # 흑백 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 검은 부분의 좌표 찾기
    black_pixels = np.where(gray <= threshold)
    coordinates = np.column_stack((black_pixels[1], black_pixels[0]))

    # 검은 부분을 흰색으로 채우기
    image[coordinates[:, 1], coordinates[:, 0]] = (255, 255, 255)

    return image
# 이미지 불러오기
image = cv2.imread('D:/test_10x10_image.png')

# 검은 부분을 하얀색으로 채우기 (픽셀 수 임계값: 100)
result = fill_black_area(image, 130)

# 결과 이미지 출력
cv2.imshow('Result', result)
cv2.waitKey(0)
cv2.destroyAllWindows()