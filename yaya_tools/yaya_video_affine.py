import sys
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
)


# Rozszerzona etykieta umożliwiająca odbieranie kliknięć myszy
class VideoLabel(QLabel):
    clicked = pyqtSignal(QPoint)

    def mousePressEvent(self, event: "QMouseEvent") -> None:
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.pos())
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Aplikacja do przetwarzania wideo")
        self.videoCapture: Optional[cv2.VideoCapture] = None
        self.timer: QTimer = QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.currentFrame: Optional[np.ndarray] = None
        self.homography: Optional[np.ndarray] = None  # Macierz przekształcenia
        self.dst_size: Optional[Tuple[int, int]] = None  # Rozmiar docelowy (szerokość, wysokość)
        self.drawing_mode: bool = False  # Flaga trybu rysowania punktów
        self.points: List[List[int]] = []  # Lista klikniętych punktów (źródłowych)
        self.playing: bool = False
        self.speedMultiplier: int = 1
        self.fps: float = 25.0

        # Main Video label
        self.main_video_label: VideoLabel = VideoLabel()
        self.main_video_label.setAlignment(Qt.AlignCenter)
        self.main_video_label.clicked.connect(self.getPoint)

        # Homography Video label
        self.homography_video_label: VideoLabel = VideoLabel()
        self.homography_video_label.setAlignment(Qt.AlignCenter)
        self.homography_video_label.clicked.connect(self.getPoint)

        # Przyciski + kontener horyzontalny
        self.openButton: QPushButton = QPushButton("Otwórz")
        self.openButton.clicked.connect(self.openFile)

        self.startStopButton: QPushButton = QPushButton("Start")
        self.startStopButton.clicked.connect(self.togglePlayback)

        self.speedComboBox: QComboBox = QComboBox()
        self.speedComboBox.addItems(["1x", "2x", "4x"])
        self.speedComboBox.currentIndexChanged.connect(self.changeSpeed)

        self.transformButton: QPushButton = QPushButton("Dodaj przekształcenie")
        self.transformButton.clicked.connect(self.activateDrawingMode)

        self.resetButton: QPushButton = QPushButton("Reset przekształcenia")
        self.resetButton.clicked.connect(self.resetTransformation)

        buttonsLayout: QHBoxLayout = QHBoxLayout()
        buttonsLayout.addWidget(self.openButton)
        buttonsLayout.addWidget(self.startStopButton)
        buttonsLayout.addWidget(self.speedComboBox)
        buttonsLayout.addWidget(self.transformButton)
        buttonsLayout.addWidget(self.resetButton)
        buttonsLayout.addStretch()  # Spacer na końcu

        # Główny layout – najpierw kontener z przyciskami, potem etykieta wideo
        mainLayout: QVBoxLayout = QVBoxLayout()
        mainLayout.addLayout(buttonsLayout)
        # Horizontal Layout inside mainLayout for two video labels
        horizontalLayout: QHBoxLayout = QHBoxLayout()
        horizontalLayout.addWidget(self.main_video_label)
        horizontalLayout.addWidget(self.homography_video_label)
        mainLayout.addLayout(horizontalLayout)

        container: QWidget = QWidget()
        container.setLayout(mainLayout)
        self.setCentralWidget(container)

    def resize_frame(self, frame: np.ndarray) -> np.ndarray:
        """Redukcja klatki do maksymalnej szerokości 1280 pikseli przy zachowaniu proporcji."""
        max_width = 800
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_w = max_width
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        return frame

    def openFile(self) -> None:
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Wybierz plik wideo", "", "Pliki wideo (*.avi *.mp4 *.mov *.mkv)"
        )
        if fileName:
            self.homography = None
            self.dst_size = None
            self.videoCapture = cv2.VideoCapture(fileName)
            ret, frame = self.videoCapture.read()
            if ret:
                frame = self.resize_frame(frame)
                self.currentFrame = frame
                height, width, _ = frame.shape
                # Ustawiamy rozmiar etykiety zgodnie z przeskalowaną klatką
                self.main_video_label.setFixedSize(width, height)
                self.displayFrame(frame)
                self.fps = self.videoCapture.get(cv2.CAP_PROP_FPS)
                if self.fps <= 0:
                    self.fps = 25
                # Nie uruchamiamy timera automatycznie
                self.timer.setInterval(int(1000 / self.fps))
                self.playing = False
                self.startStopButton.setText("Start")
                print("Plik wczytany. Kliknij Start aby rozpocząć odtwarzanie.")

    def nextFrameSlot(self) -> None:
        if self.videoCapture is not None:
            ret, frame = self.videoCapture.read()
            if ret:
                frame = self.resize_frame(frame)
                self.currentFrame = frame
                self.displayFrame(frame)
                # Pomijamy dodatkowe klatki wg. wybranego mnożnika
                for _ in range(self.speedMultiplier - 1):
                    ret_skip, _ = self.videoCapture.read()
                    if not ret_skip:
                        self.videoCapture.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        break
            else:
                # Po zakończeniu odtwarzania wideo, wracamy do początku
                self.videoCapture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def displayFrame(self, frame: np.ndarray) -> None:
        # Jeśli macierz homografii istnieje, przekształcamy klatkę
        homography_frame = frame
        if self.homography is not None and self.dst_size is not None:
            homography_frame = cv2.warpPerspective(frame, self.homography, self.dst_size)

        # Jeśli jesteśmy w trybie rysowania, nakładamy punkty i linie, by pokazać powstający poligon
        if self.drawing_mode and self.points:
            # Rysuj punkty
            for pt in self.points:
                cv2.circle(frame, (int(pt[0]), int(pt[1])), 5, (0, 0, 255), -1)

            # Rysuj linie między kolejnymi punktami
            if len(self.points) > 1:
                for i in range(len(self.points) - 1):
                    pt1 = (int(self.points[i][0]), int(self.points[i][1]))
                    pt2 = (int(self.points[i + 1][0]), int(self.points[i + 1][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
                # Jeśli użytkownik kliknął 4 punkty, zamykamy poligon
                if len(self.points) == 4:
                    pt1 = (int(self.points[3][0]), int(self.points[3][1]))
                    pt2 = (int(self.points[0][0]), int(self.points[0][1]))
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 2)

        # Base frame : Convert and display
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb.shape
        bytesPerLine = 3 * width
        qImg = QImage(rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.main_video_label.setPixmap(QPixmap.fromImage(qImg))

        # Homography frame : Convert and display
        rgb_homography = cv2.cvtColor(homography_frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb_homography.shape
        bytesPerLine = 3 * width
        qImg_homography = QImage(rgb_homography.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.homography_video_label.setPixmap(QPixmap.fromImage(qImg_homography))

    def activateDrawingMode(self) -> None:
        # Aktywacja trybu rysowania – resetujemy listę punktów
        self.drawing_mode = True
        self.points = []
        print("Kliknij 4 punkty w kolejności: Lewy dolny, Prawy dolny, Prawy górny, Lewy górny.")

    def getPoint(self, pos: QPoint) -> None:
        if self.drawing_mode:
            self.points.append([pos.x(), pos.y()])
            print(f"Zarejestrowano punkt: {pos.x()}, {pos.y()}.")
            if len(self.points) == 4:
                self.drawing_mode = False
                self.computeHomography()
                # Po obliczeniu homografii czyścimy punkty, aby nie były rysowane na przetworzonym obrazie
                self.points = []
                print("Obliczono macierz przekształcenia.")

    def computeHomography(self) -> None:
        pts = np.array(self.points, dtype="float32")
        # Zakładamy, że punkty są w kolejności:
        # [0] Lewy dolny, [1] Prawy dolny, [2] Prawy górny, [3] Lewy górny
        width_bottom = np.linalg.norm(pts[1] - pts[0])
        width_top = np.linalg.norm(pts[2] - pts[3])
        maxWidth = int(max(width_bottom, width_top))
        height_left = np.linalg.norm(pts[0] - pts[3])
        height_right = np.linalg.norm(pts[1] - pts[2])
        maxHeight = int(max(height_left, height_right))
        self.dst_size = (maxWidth, maxHeight)
        dst = np.array(
            [
                [0, maxHeight],  # Dolny lewy
                [maxWidth, maxHeight],  # Dolny prawy
                [maxWidth, 0],  # Górny prawy
                [0, 0],  # Górny lewy
            ],
            dtype="float32",
        )
        self.homography = cv2.getPerspectiveTransform(pts, dst)

    def resetTransformation(self) -> None:
        """Resetuje macierz przekształcenia."""
        self.homography = None
        self.dst_size = None
        print("Resetowano przekształcenie.")

    def togglePlayback(self) -> None:
        if self.playing:
            self.timer.stop()
            self.playing = False
            self.startStopButton.setText("Start")
        else:
            self.timer.start(int(1000 / self.fps))
            self.playing = True
            self.startStopButton.setText("Stop")

    def changeSpeed(self) -> None:
        # Aktualizacja mnożnika na podstawie wybranej wartości w comboboxie
        speed_text = self.speedComboBox.currentText()
        self.speedMultiplier = int(speed_text.replace("x", ""))
        print(f"Wybrana prędkość: {self.speedMultiplier}x")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
