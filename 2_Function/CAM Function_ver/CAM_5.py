import cv2
import numpy as np

def remapping_image(frame):
        # 왜곡 계수 설정
        # k1, k2, k3 = 0.02, 0.0, 0.0 # 배럴 왜곡
        k1, k2, k3 = -0.004, 0.0, 0.0  # 핀큐션 왜곡
        '''
        k1은 주 왜곡 효과를 조절하는 계수 값이 양수일때 이미지가 중심을 기준으로 원형으로 퍼짐, 
        음수일때 이미지가 중심을 기준으로 원형으로 좁아짐
        k2는 k1의 제곱에 비례하는 왜곡 효과
        k3는 k1의 세제곱에 비례하는 왜곡 효과 거의 사용하지 않아 0으로 설정
        '''
        rows, cols = frame.shape[:2]
        # 매핑 배열 생성
        mapy, mapx = np.indices((rows, cols), dtype=np.float32)
        # 중앙점 좌표로 -1~1 정규화 및 극좌표 변환
        mapx = 2 * mapx / (cols - 1) - 1
        mapy = 2 * mapy / (rows - 1) - 1
        r, theta = cv2.cartToPolar(mapx, mapy)
        # 방사 왜곡 변영 연산
        ru = r * (1 + k1 * (r ** 2) + k2 * (r ** 4) + k3 * (r ** 6))
        # 직교좌표 및 좌상단 기준으로 복원
        mapx, mapy = cv2.polarToCart(ru, theta)
        mapx = ((mapx + 1) * cols - 1) / 2
        mapy = ((mapy + 1) * rows - 1) / 2
        # 리매핑
        map_image = cv2.remap(frame, mapx, mapy, cv2.INTER_LINEAR)
        return map_image

# 이미지 불러오기
image_origin = cv2.imread('./WIN_20230801_13_08_40_Pro.jpg')
image = remapping_image(image_origin.copy())

# 이미지 전처리 (그레이스케일 변환)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 이미지 크기 및 중심 좌표 계산
height, width, _ = image.shape
center_x = int(width / 2)
center_y = int(height / 2)

# 경계선 감지 (Canny edge detection)
edges = cv2.Canny(gray, 40, 95)

# 윤곽선 찾기
contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# 윤곽선 색상 지정
contour_color = (0, 0, 255)  # 빨간색 (BGR 순서)

# 필터링할 윤곽선 저장 리스트
filtered_contours = []

# 윤곽선을 순회하며 면적이 일정 크기 이상인 윤곽선만 저장
min_contour_area = 60  # 예시로 100 픽셀로 지정
max_contour_area = 70
for contour in contours:
    area = cv2.contourArea(contour)
    if min_contour_area < area < max_contour_area:
        filtered_contours.append(contour)

# 윤곽선 그리기
cv2.drawContours(image, filtered_contours, -1, contour_color, 2)

# 꼭지점 좌표 추출
location_nparr=[]
for contour in filtered_contours:
    # 윤곽선 근사화
    epsilon = 0.02 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)

    # 꼭지점 좌표에 파란 점 찍기
    for point in approx:
        x, y = point[0]
        location_nparr.append([x,y])
        cv2.circle(image, (x, y), 5, (255, 0, 0), -1)
location = np.array(location_nparr, np.float32)
location2 = np.array([[0, 0], [0, height], [width, height], [width, 0]], np.float32)
pers = cv2.getPerspectiveTransform(location, location2)
dst = cv2.warpPerspective(image_origin, pers, (width,height))
dst=cv2.resize(dst,(1000,800))
# 이미지 크기 및 중심 좌표 계산
height, width, _ = image.shape
cell_width = int(width / 10)
cell_height = int(height / 10)

# # 구분선 그리기
# for i in range(1, 10):
#     x = i * cell_width
#     y = i * cell_height
#     cv2.line(dst, (x, 0), (x, height), (255, 0, 255), 1)
#     cv2.line(dst, (0, y), (width, y), (255, 0, 255), 1)

# 영역 이미지 출력
cv2.imshow('Contours', image)
cv2.imshow('next',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()