import cv2
import numpy as np

# 마우스 클릭 이벤트 핸들러
def mouse_click(event, x, y, flags, param):
    global points, num_points, selecting

    if event == cv2.EVENT_LBUTTONDOWN:  # 마우스 왼쪽 버튼 클릭 이벤트
        points.append((x, y))  # 클릭한 좌표 저장
        selecting = True

        num_points += 1

        # 4개의 꼭지점을 모두 선택한 경우
        if num_points == 4:
            print("Selected Points:", points)
            # 이벤트 핸들러 비활성화
            cv2.setMouseCallback('Select Points', lambda *args: None)

            # 이미지 창 크기 조정
            adjust_window_size(image)

def remapping_image(image):
        # 왜곡 계수 설정
        # k1, k2, k3 = 0.02, 0.0, 0.0 # 배럴 왜곡
        k1, k2, k3 = -0.01, 0.0, 0.0  # 핀큐션 왜곡
        '''
        k1은 주 왜곡 효과를 조절하는 계수 값이 양수일때 이미지가 중심을 기준으로 원형으로 퍼짐, 
        음수일때 이미지가 중심을 기준으로 원형으로 좁아짐
        k2는 k1의 제곱에 비례하는 왜곡 효과
        k3는 k1의 세제곱에 비례하는 왜곡 효과 거의 사용하지 않아 0으로 설정
        '''
        rows, cols = image.shape[:2]
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
        map_image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
        return map_image

def adjust_window_size(image):
    # 이미지 창 크기 조정
    height, width = image.shape[:2]

    # 선택한 꼭지점 좌표
    pts = np.array(points, np.float32)
    dst_pts = np.array([[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]], np.float32)

    # 변환 행렬 계산
    matrix = cv2.getPerspectiveTransform(pts, dst_pts)

    # 이미지 창 크기 조정
    warped_image = cv2.warpPerspective(image, matrix, (width, height))

    # 이미지 출력
    cv2.imshow('Adjusted Image', warped_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 이미지 불러오기
image = cv2.imread('D:/test_10x10_image.png')
image = remapping_image(image)

# 윈도우 생성 및 이미지 출력
cv2.imshow('Select Points', image)

# 클릭한 좌표를 저장할 리스트와 꼭지점 선택 횟수 변수
points = []
num_points = 0
selecting = False

# 마우스 클릭 이벤트 핸들러 등록
cv2.setMouseCallback('Select Points', mouse_click)

while True:
    # 이미지 출력
    img_show = image.copy()
    
    if selecting:
        # 선택 중인 경우 빨간 점 표시
        for point in points:
            cv2.circle(img_show, point, 5, (0, 0, 255), -1)
    
    cv2.imshow('Select Points', img_show)
    
    # 키 입력 대기
    key = cv2.waitKey(1) & 0xFF
    
    # 'q'를 누르면 종료
    if key == ord('q'):
        break

cv2.destroyAllWindows()