import sys

import cv2
import numpy as np
from PyQt5.QtCore import QPoint, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
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

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(event.pos())
        super().mousePressEvent(event)


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Aplikacja do przetwarzania wideo")
        self.videoCapture = None
        self.timer = QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.currentFrame = None
        self.homography = None  # Macierz przekształcenia
        self.dst_size = None  # Rozmiar docelowy (szerokość, wysokość)
        self.drawing_mode = False  # Flaga trybu rysowania punktów
        self.points = []  # Lista klikniętych punktów (źródłowych)

        # Interfejs – etykieta do wyświetlania klatek
        self.videoLabel = VideoLabel()
        self.videoLabel.setAlignment(Qt.AlignCenter)
        self.videoLabel.clicked.connect(self.getPoint)

        # Przyciski + kontener horyzontalny
        self.openButton = QPushButton("Otwórz")
        self.openButton.clicked.connect(self.openFile)

        self.transformButton = QPushButton("Dodaj przekształcenie")
        self.transformButton.clicked.connect(self.activateDrawingMode)

        self.resetButton = QPushButton("Reset przekształcenia")
        self.resetButton.clicked.connect(self.resetTransformation)

        buttonsLayout = QHBoxLayout()
        buttonsLayout.addWidget(self.openButton)
        buttonsLayout.addWidget(self.transformButton)
        buttonsLayout.addWidget(self.resetButton)
        buttonsLayout.addStretch()  # Spacer na końcu

        # Główny layout – najpierw kontener z przyciskami, potem etykieta wideo
        mainLayout = QVBoxLayout()
        mainLayout.addLayout(buttonsLayout)
        mainLayout.addWidget(self.videoLabel)

        container = QWidget()
        container.setLayout(mainLayout)
        self.setCentralWidget(container)

    def resize_frame(self, frame):
        """Redukcja klatki do maksymalnej szerokości 1280 pikseli przy zachowaniu proporcji."""
        max_width = 1280
        h, w = frame.shape[:2]
        if w > max_width:
            scale = max_width / w
            new_w = max_width
            new_h = int(h * scale)
            frame = cv2.resize(frame, (new_w, new_h))
        return frame

    def openFile(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Wybierz plik wideo", "", "Pliki wideo (*.avi *.mp4 *.mov *.mkv)"
        )
        if fileName:
            # Reset przekształcenia przy otwarciu nowego pliku
            self.homography = None
            self.dst_size = None
            self.videoCapture = cv2.VideoCapture(fileName)
            ret, frame = self.videoCapture.read()
            if ret:
                frame = self.resize_frame(frame)
                self.currentFrame = frame
                height, width, _ = frame.shape
                # Ustawiamy rozmiar etykiety zgodnie z przeskalowaną klatką
                self.videoLabel.setFixedSize(width, height)
                self.displayFrame(frame)
                fps = self.videoCapture.get(cv2.CAP_PROP_FPS)
                if fps <= 0:
                    fps = 25
                self.timer.start(int(1000 / fps))

    def nextFrameSlot(self):
        if self.videoCapture is not None:
            ret, frame = self.videoCapture.read()
            if ret:
                frame = self.resize_frame(frame)
                self.currentFrame = frame
                # Jeśli macierz homografii istnieje, przekształcamy klatkę
                if self.homography is not None and self.dst_size is not None:
                    frame = cv2.warpPerspective(frame, self.homography, self.dst_size)
                self.displayFrame(frame)
            else:
                # Po zakończeniu odtwarzania wideo, wracamy do początku
                self.videoCapture.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def displayFrame(self, frame):
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

        # Konwersja z BGR do RGB i przekształcenie do QImage
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channel = rgb.shape
        bytesPerLine = 3 * width
        qImg = QImage(rgb.data, width, height, bytesPerLine, QImage.Format_RGB888)
        self.videoLabel.setPixmap(QPixmap.fromImage(qImg))

    def activateDrawingMode(self):
        # Aktywacja trybu rysowania – resetujemy listę punktów
        self.drawing_mode = True
        self.points = []
        print("Kliknij 4 punkty w kolejności: Lewy dolny, Prawy dolny, Prawy górny, Lewy górny.")

    def getPoint(self, pos):
        if self.drawing_mode:
            self.points.append([pos.x(), pos.y()])
            print(f"Zarejestrowano punkt: {pos.x()}, {pos.y()}.")
            if len(self.points) == 4:
                self.drawing_mode = False
                self.computeHomography()
                # Po obliczeniu homografii czyścimy punkty, aby nie były rysowane na przetworzonym obrazie
                self.points = []
                print("Obliczono macierz przekształcenia.")

    def computeHomography(self):
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

    def resetTransformation(self):
        """Resetuje macierz przekształcenia."""
        self.homography = None
        self.dst_size = None
        print("Resetowano przekształcenie.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
