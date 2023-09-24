import cv2
import ctypes
import numpy as np

def draw_contours_of_color(image, target_color):
    # 이미지를 BGR 형식으로 로드
    img_bgr = cv2.imread(image)

    # 이미지를 RGB 형식으로 변환
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # RGB 이미지를 HSV 형식으로 변환
    img_hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # 목표 색상의 HSV 범위를 정의합니다. 여기서는 #ACACAC 색상을 사용합니다.
    lower_color = np.array([0, 0, 162])
    upper_color = np.array([0, 0, 172])

    # HSV 이미지에서 목표 색상을 필터링합니다.
    mask = cv2.inRange(img_hsv, lower_color, upper_color)

    # 윤곽선을 찾습니다.
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 윤곽선을 이미지에 그립니다.
    for contour in contours:
        cv2.drawContours(img_rgb, [contour], -1, (255, 0, 0), 2)  # 파란색으로 윤곽선 그리기

    # 이미지를 BGR 형식으로 변환하여 반환
    result_img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return result_img_bgr

def get_monitor_resolution():
    user32 = ctypes.windll.user32
    return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)


if __name__ == "__main__":
    input_image = './5x10_Origin_Image.jpg'# 이미지 파일 경로를 지정해주세요.
    output_image = draw_contours_of_color(input_image, (172, 172, 172))  # #ACACAC 색상

    # 모니터 해상도를 얻습니다.
    screen_width, screen_height = get_monitor_resolution()

    # 이미지 크기를 얻습니다.
    image_height, image_width, _ = output_image.shape

    # 스케일링 팩터를 계산합니다.
    scale_factor = min(screen_width / image_width, screen_height / image_height)

    # 이미지가 화면보다 크다면 크기를 조정합니다.
    if scale_factor < 1:
        approx_frame = cv2.resize(output_image, None, fx=scale_factor, fy=scale_factor)

    # 창 이름을 설정합니다.
    window_name = "approx_pos"

    # 전체화면 모드로 이미지를 보여줍니다.
    cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    cv2.imshow(window_name, output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
