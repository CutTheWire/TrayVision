import cv2
import ctypes

def resize_to_monitor(image):
    # 모니터의 해상도 정보 얻기
    user32 = ctypes.windll.user32
    screen_width = user32.GetSystemMetrics(0)
    screen_height = user32.GetSystemMetrics(1)

    # 이미지 크기 얻기
    image_height, image_width, _ = image.shape

    # 모니터에 맞게 이미지 리사이즈
    scale_factor = min(screen_width / image_width, screen_height / image_height)
    resized_image = cv2.resize(image, None, fx=scale_factor, fy=scale_factor)

    return resized_image

def apply_edge_detection(image):
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Canny 에지 검출
    edges = cv2.Canny(gray, threshold1=200, threshold2=200)

    # 원본 이미지에 에지 표시
    result = cv2.bitwise_and(image, image, mask=edges)

    return result

# 이미지 경로
image_path = './WIN_2023.jpg'

image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# 모니터 크기로 이미지 리사이즈
resized_image = apply_edge_detection(resize_to_monitor(image))

# 리사이즈된 이미지 출력
cv2.imshow('Resized Image', resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()