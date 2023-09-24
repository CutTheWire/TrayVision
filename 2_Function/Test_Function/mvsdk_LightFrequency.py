import mvsdk

# 카메라 초기화
DevList = mvsdk.CameraEnumerateDevice()
if len(DevList) < 1:
    print("카메라를 찾을 수 없습니다!")
else:
    DevInfo = DevList[0]
    hCamera = mvsdk.CameraInit(DevInfo, -1, -1)

    if hCamera != 0:
        # CameraGetLightFrequency 함수 호출
        status = mvsdk.CameraGetLightFrequency(hCamera)
        print(status)
        mvsdk.CameraSetLightFrequency(hCamera, 10)
        status = mvsdk.CameraGetLightFrequency(hCamera)
        print(status)
        # 카메라 해제
        mvsdk.CameraUnInit(hCamera)
    else:
        print("카메라 초기화 실패")
