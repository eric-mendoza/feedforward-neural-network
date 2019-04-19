# Canvas
#

from PyQt5.QtWidgets import QApplication, QMainWindow, QMenuBar, QMenu, QAction, QFileDialog
from PyQt5.QtGui import QIcon, QImage, QPainter, QPen, QPixmap
from PyQt5.QtCore import Qt, QPoint
import sys


#  Create the main window for the painter
class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        # Set initial values for window
        top = 200
        left = 200
        width = 800
        height = 800
        self.setWindowTitle("Painter")
        self.setGeometry(top, left, width, height)

        # Crete canvas for drawing
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)

        self.drawing = False
        self.brushSize = 80
        self.brushColor = Qt.black
        self.lastPoint = QPoint()

        # Create main menu
        main_menu = self.menuBar()
        file_menu = main_menu.addMenu("File")
        brush_menu = main_menu.addMenu("Brush Size")
        brush_color = main_menu.addMenu("Brush Color")

        # Add options to 'File' menu
        save_action = QAction("Save", self)
        save_action.setShortcut("Ctrl+S")
        file_menu.addAction(save_action)
        save_action.triggered.connect(self.save)

        clear_action = QAction("Clear", self)
        clear_action.setShortcut("Ctrl+C")
        file_menu.addAction(clear_action)
        clear_action.triggered.connect(self.clear)

        # Add options to 'Brush' menu
        threepx_action = QAction("3px", self)
        threepx_action.setShortcut("Ctrl+3")
        brush_menu.addAction(threepx_action)

        fivepx_action = QAction("5px", self)
        fivepx_action.setShortcut("Ctrl+5")
        brush_menu.addAction(fivepx_action)

        sevenpx_action = QAction("7px", self)
        sevenpx_action.setShortcut("Ctrl+7")
        brush_menu.addAction(sevenpx_action)

        ninepx_action = QAction("9px", self)
        ninepx_action.setShortcut("Ctrl+9")
        brush_menu.addAction(ninepx_action)

        # Add options to 'Brush Color' menu
        black_action = QAction("Black", self)
        black_action.setShortcut("Ctrl+B")
        brush_color.addAction(black_action)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True
            self.lastPoint = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton & self.drawing:
            painter = QPainter(self.image)
            painter.setPen(QPen(self.brushColor, self.brushSize, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            painter.drawLine(self.lastPoint, event.pos())
            self.lastPoint = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button == Qt.LeftButton:
            self.drawing = False

    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect(), self.image, self.image.rect())

    def save(self):
        file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "BMP(*.bmp);; ALL Files(*.*)")
        if file_path == "":
            return
        new = self.image.scaled(28, 28)  # Escalar la imagen
        new.save(file_path)

    def clear(self):
        self.image.fill(Qt.white)
        self.update()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    app.exec()
