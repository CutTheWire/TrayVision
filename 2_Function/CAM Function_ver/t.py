from PyQt5.QtWidgets import *

app = QApplication([])

# 모니터 갯수 반환
app.desktop().screenNumber()

# 주 모니터 선택
screen = app.primaryScreen()
# 모니터 이름 (\\.\DISPLAY1)
print(screen.name())
# 해상도 ( 작업표시줄을 제외한 영역을 반환할 경우 아래 함수 사용 )
print(screen.geometry())
print(screen.availableGeometry())

# 다중 모니터 해상도 구하기
monitorCount = app.desktop().screenCount()
for i in range(monitorCount):
    print(app.desktop().availableGeometry(i))
    # 모니터 이름
    print(app.desktop().screen(i).screen().name())

# 주 모니터의 가로와 세로 크기 출력
print("Primary Monitor:")
print("Width:", screen.size().width())
print("Height:", screen.size().height())

# 다중 모니터 해상도 구하기
monitorCount = app.desktop().screenCount()
for i in range(monitorCount):
    monitor = app.desktop().screen(i)
    print("Monitor", i + 1)
    print("Width:", monitor.size().width())
    print("Height:", monitor.size().height())

app.exec_()
