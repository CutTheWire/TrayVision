import numpy as np
import cv2

# 카메라 열기
cap = cv2.VideoCapture(2)  # 0은 기본 카메라 장치 번호, 더 많은 카메라가 있는 경우에는 1, 2, 등을 시도해볼 수 있습니다.

# 프레임 수 구하기
fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(100 / fps)

# 배경 제거 객체 생성
fgbg = cv2.createBackgroundSubtractorMOG2()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # 배경 제거 마스크 계산
    fgmask = fgbg.apply(frame)

    cv2.imshow('frame', frame)
    cv2.imshow('bgsub', fgmask)
    
    if cv2.waitKey(delay) & 0xff == 27:
        break

cap.release()
cv2.destroyAllWindows()
