import subprocess

allowed_mac_address = "8C-17-59-F1-21-47"  # 허용된 MAC 주소

def get_physical_address():
    result = subprocess.run(["ipconfig", "/all"], capture_output=True, text=True)
    output = result.stdout
    # 물리적 주소(MAC 주소)를 찾아 반환합니다.
    mac_addresses = re.findall(r"Physical Address[\. ]+: ([\w-]+)", output)
    if mac_addresses:
        return mac_addresses[0].replace("-", ":")  # MAC 주소의 구분자를 ":"로 변경
    else:
        return None

def check_mac_address():
    physical_address = get_physical_address()
    if physical_address == allowed_mac_address:
        return True
    else:
        return False

def capture_photo():
    if not check_mac_address():
        print("허용되지 않은 PC입니다.")
        return

    # 이하 코드 생략