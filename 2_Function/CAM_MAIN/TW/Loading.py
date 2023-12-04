import base64
from PyQt5.QtWidgets import QApplication, QWidget
app = QApplication([])
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint
from screeninfo import get_monitors
from bin.base64_data import logo_image_base64
class LoadingScreen(QWidget):
    def __init__(self) -> None:
        """
        LoadingScreen 클래스의 초기화 함수입니다.
        로딩 화면을 설정하고, 이미지를 표시하는 등의 역할을 합니다.
        """
        super().__init__()
        self.setWindowTitle(" ")
        self.setWindowFlags(Qt.WindowStaysOnTopHint | Qt.FramelessWindowHint | Qt.WindowTransparentForInput)
        self.resize(650, 400)  # 창 크기 설정

        # 로딩 중 이미지를 표시할 QLabel
        self.loading_label = QLabel(self)

        # 로딩 이미지를 base64로 변경하여 QLabel에 설정
        base64_image = logo_image_base64
        image_data = base64.b64decode(base64_image)
        image = QImage.fromData(image_data)
        pixmap = QPixmap.fromImage(image)
        self.loading_label.setPixmap(pixmap)

        # 수직 레이아웃을 사용하여 이미지를 중앙에 배치
        layout = QVBoxLayout(self)
        layout.addWidget(self.loading_label)
        layout.setAlignment(Qt.AlignCenter)

        self.loading_label.setAutoFillBackground(False)  # 배경 채우기 비활성화
        self.loading_label.setAttribute(Qt.WA_TranslucentBackground)  # 배경을 투명하게 만듭니다

        # 중앙 좌표를 계산
        center_point = self.calculateCenterPoint()
        self.moveCenter(center_point)

    def calculateCenterPoint(self) -> QPoint:
        """
        화면의 중앙 좌표를 계산하는 함수입니다.
        연결된 모니터의 크기를 바탕으로 중앙 좌표를 계산하여 반환합니다.
        """
        monitors = get_monitors()
        x = 0
        y = 0
        if monitors:
            monitor = monitors[0]
            x += monitor.width
            y += monitor.height
            center_x = x // 2
            center_y = y // 2
            return QPoint(center_x, center_y)
        else:
            return QPoint(x, y)

    def moveCenter(self, point: QPoint) -> None:
        """
        로딩 화면을 화면의 중앙으로 이동시키는 함수입니다.
        계산된 중앙 좌표를 바탕으로 로딩 화면의 위치를 이동시킵니다.
        """
        qr = self.frameGeometry()
        qr.moveCenter(point)
        self.move(qr.topLeft())

    def show(self) -> None:
        """
        로딩 화면을 표시하는 함수입니다.
        부모 클래스의 show 메서드를 호출하여 로딩 화면을 표시합니다.
        """
        super().show()

    def close(self) -> None:
        """
        로딩 화면을 닫는 함수입니다.
        부모 클래스의 close 메서드를 호출하여 로딩 화면을 닫습니다.
        """
        super().close()
