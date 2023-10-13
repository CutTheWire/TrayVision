import os
import pickle

data = {
    '10x10': 46.4,
    '5x10': 47.7,
    'S_10x10': 59.0,
    '200-1': 18.1,
    'aaa': 111.0
}

# 데이터를 바이너리 파일에 쓰기
document_folder = os.path.expanduser("~/Documents")
file_path = os.path.join(document_folder, "TW", "settings.bin")

with open(file_path, 'wb') as binary_file:
    pickle.dump(data, binary_file)

# 바이너리 파일에서 데이터를 읽어 딕셔너리로 변환
with open(file_path, 'rb') as binary_file:
    loaded_data = pickle.load(binary_file)

print(loaded_data)
