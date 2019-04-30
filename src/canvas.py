from PyQt5.QtWidgets import QApplication, QMainWindow, QMenuBar, QMenu, QAction, QFileDialog, QMessageBox
from PyQt5.QtGui import QIcon, QImage, QPainter, QPen, QPixmap
from PyQt5.QtCore import Qt, QPoint
import sys
import src.data_loader as loader
import src.network as network


#  Create the main window for the painter
class Window(QMainWindow):
    def __init__(self, neural_network):
        super().__init__()

        # Save the network for future predictions
        self.network = neural_network

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
        self.brushSize = 60
        self.brushColor = Qt.black
        self.lastPoint = QPoint()

        # Create main menus
        main_menu = self.menuBar()
        file_menu = main_menu.addMenu("File")
        brush_menu = main_menu.addMenu("Brush Size")
        brush_color = main_menu.addMenu("Brush Color")
        predict_menu = main_menu.addMenu("Predict")

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

        # Add options to 'Predict' menu
        predict_action = QAction("Guess number", self)
        predict_action.setShortcut("Ctrl+P")
        predict_menu.addAction(predict_action)
        predict_action.triggered.connect(self.guess)

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

    def guess(self):
        # Obtener imagen
        image = self.image.scaled(28, 28)  # Scale image
        n = loader.qimage_to_ndarray(image)
        drawing, percentage = self.network.evaluate_drawing(n)
        message = ("I'm %s sure it's a %s!" % (percentage[0], loader.image_classifier(int(drawing))))
        QMessageBox.about(self, "Result", message)


if __name__ == '__main__':
    # Load Data
    train, cv, test = loader.load_data()

    # Create neural network
    net = network.Network([784, 30, 10])

    # training_data, epochs, mini_batch_size, eta, test_data
    net.sgd(train, 10, 10, 3.0, test)

    app = QApplication(sys.argv)
    window = Window(net)
    window.show()
    app.exec()
