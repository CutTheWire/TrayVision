import os
import cv2

def get_unique_filename(folder, base_filename, extension):
    # 중복되지 않는 파일 이름을 생성하기 위한 함수
    filename = f"{base_filename}{extension}"
    count = 1
    while os.path.exists(os.path.join(folder, filename)):
        filename = f"{base_filename}_{count}{extension}"
        count += 1
    return filename

documents_folder = os.path.join(os.path.expanduser("~"), "Documents")
main_folder = os.path.join(documents_folder, "TWCV")

# 폴더가 이미 존재하는지 확인 후 생성
if not os.path.exists(main_folder):
    os.mkdir(main_folder)

frame_path = "C:/tinywave/camera_disconnected.png"

# 이미지를 읽어옴
frame = cv2.imread(frame_path)

filename = get_unique_filename(main_folder, "photo", ".jpg")
photo_path = os.path.join(main_folder, filename)

# 이미지를 저장
cv2.imwrite(photo_path, frame)
print(f"{photo_path}에 사진을 저장했습니다.")
