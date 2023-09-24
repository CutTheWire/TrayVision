import cv2
from pygrabber.dshow_graph import FilterGraph

def get_available_cameras() :

    devices = FilterGraph().get_input_devices()

    available_cameras = {}

    for device_index, device_name in enumerate(devices):
        available_cameras[device_index] = device_name

    return available_cameras

camera_names = get_available_cameras()
print(camera_names)

for i in camera_names:
    if  camera_names[i] == 'usb-webcam':

        cap = cv2.VideoCapture(i)
    else:
        cap = cv2.VideoCapture(0)

# 카메라가 정상적으로 열렸는지 확인
if not cap.isOpened():
    print("카메라를 열 수 없습니다.")
else:
    while True:
        # 프레임 읽기
        ret, frame = cap.read()

        if not ret:
            print("프레임을 읽을 수 없습니다.")
            break

        # 프레임 표시
        cv2.imshow("CAM", frame)

        # 'q' 키를 누르면 루프 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 작업이 완료되면 카메라 해제
    cap.release()
    cv2.destroyAllWindows()
