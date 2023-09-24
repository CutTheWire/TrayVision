import cv2

def find_contours(image_path):
    # 이미지 불러오기
    image = cv2.imread(image_path)
    
    image = cv2.bitwise_not(image)

    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 이진화
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    # 윤곽선 찾기
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 윤곽선 그리기
    cv2.drawContours(image, contours, -1, (0, 255, 0), 2)

    # 결과 이미지 출력
    cv2.imshow('Contours', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 윤곽선을 찾을 이미지 경로 설정
image_path = 'D:/test_10x10_image.png'

# 윤곽선 찾기 수행
find_contours(image_path)
