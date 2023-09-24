import cv2
import numpy as np
import sys

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

# 이미지 전처리
def preprocessing_image(image, n, m):
    # 그레이스케일 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # 경계선 감지 (Canny edge detection)
    edges = cv2.Canny(gray, n, m)
    
    # 윤곽선 찾기
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    return contours, _

# 꼭지점 좌표 추출
def coordinate_extraction(contours):
    for contour in contours:
        # 윤곽선 근사화
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        return approx

# 이미지 불러오기
image = cv2.imread('D:/test_10x10_image.png')
if image is None:
    print("이미지 파일을 찾을 수 없습니다.")
    sys.exit()
image = cv2.resize(image, dsize=(640, 480), interpolation=cv2.INTER_AREA)
image = remapping_image(image)

# 이미지 크기 및 중심 좌표 계산
height, width, _ = image.shape

# 초기 n, m 값 설정
n = 0
m = 55

# 꼭지점 좌표 추출용 배열
location_nparr = []
approx = None
while approx is None or len(approx) != 4:
    # 이미지 전처리 및 좌표 추출
    contours, _ = preprocessing_image(image, n, m)
    approx = coordinate_extraction(contours)
    
    # n, m 값 조건 확인
    if len(approx) != 4:
        # n, m 값 조건을 충족하지 않을 경우 수정
        n += 1
        m += 1
        
        # n, m 값 조건 추가 확인
        if n >= 255 or m >= 255:
            dst = image
            break
    else:
        # n, m 값 확인
        print("Final n, m values:", n, m)

        # 좌표 정렬
        approx = np.array(sorted(approx, key=lambda x: x[0][0] + x[0][1]))

        # 꼭지점 좌표에 파란 점 찍기
        for point in approx:
            x, y = point[0]
            location_nparr.append([x, y])

        location = np.array(location_nparr, np.float32)
        location2 = np.array([[width, height], [width, 0], [0, height], [0, 0]], np.float32)
        pers = cv2.getPerspectiveTransform(location, location2)
        dst = cv2.warpPerspective(image, pers, (width, height))

        # dst 이미지 방향 보정
        dst_corrected = cv2.transpose(dst)
        dst_corrected = cv2.flip(dst_corrected, 1)

        # 180도 회전
        dst = cv2.rotate(dst_corrected, cv2.ROTATE_90_CLOCKWISE)


# 이미지 크기 및 중심 좌표 계산
height, width, _ = image.shape
cell_width = int(width / 10)
cell_height = int(height / 10)

# 구분선 그리기
for i in range(1, 10):
    x = i * cell_width
    y = i * cell_height
    cv2.line(dst, (x, 0), (x, height), (255, 0, 255), 1)
    cv2.line(dst, (0, y), (width, y), (255, 0, 255), 1)

# 영역 이미지 출력
cv2.imshow('Contours', image)
cv2.imshow('next',dst)
cv2.waitKey(0)
cv2.destroyAllWindows()