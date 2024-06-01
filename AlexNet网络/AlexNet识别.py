import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, QLabel
from PyQt5.QtCore import Qt, QPoint
from PyQt5.QtGui import QPainter, QPen, QImage, QPixmap

## 该程序实现一个手写数字识别系统，包括训练好的AlexNet模型的加载和一个用于手写数字识别的GUI应用。
## 技术方法包括使用PyTorch定义和加载AlexNet模型，并通过PyQt5构建图形用户界面，允许用户在画布上手写数字并进行预测。

# 定义AlexNet模型
class AlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 第二层卷积
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            # 第三层卷积
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 第四层卷积
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # 第五层卷积
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)  # 特征提取
        x = x.view(x.size(0), 256 * 6 * 6)  # 展平
        x = self.classifier(x)  # 分类
        return x

# 自定义画布组件，用户可以在上面绘制手写数字
class DrawWidget(QWidget):
    def __init__(self, parent=None):
        super(DrawWidget, self).__init__(parent)
        self.setFixedSize(224, 224)
        self.setStyleSheet("background-color: white;")
        self.image = QImage(self.size(), QImage.Format_RGB32)
        self.image.fill(Qt.white)
        self.drawing = False  # 是否正在绘制
        self.last_point = QPoint()  # 上一个点的位置

    def paintEvent(self, event):
        canvas_painter = QPainter(self)
        canvas_painter.drawImage(self.rect(), self.image, self.image.rect())

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = True  # 开始绘制
            self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.LeftButton & self.drawing:
            painter = QPainter(self.image)
            pen = QPen(Qt.black, 15, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())
            self.last_point = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drawing = False  # 结束绘制

    def clear(self):
        self.image.fill(Qt.white)  # 清除画布
        self.update()

    def get_image(self):
        return self.image

# 主窗口类，包含绘图画布和按钮
class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle("手写数字识别")
        self.setGeometry(100, 100, 300, 400)
        self.draw_widget = DrawWidget()  # 绘图组件

        self.label = QLabel()
        self.label.setFixedSize(224, 224)

        self.clear_button = QPushButton("清除")
        self.clear_button.clicked.connect(self.clear_canvas)  # 连接清除按钮的事件

        self.predict_button = QPushButton("识别")
        self.predict_button.clicked.connect(self.predict_digit)  # 连接识别按钮的事件

        layout = QVBoxLayout()
        layout.addWidget(self.draw_widget)
        layout.addWidget(self.label)
        button_layout = QHBoxLayout()
        button_layout.addWidget(self.clear_button)
        button_layout.addWidget(self.predict_button)
        layout.addLayout(button_layout)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        self.model = AlexNet(num_classes=10)  # 实例化AlexNet模型
        self.model.load_state_dict(torch.load('E:\\Python-projects\\alexnet_mnist.pth', map_location=torch.device('cpu')))  # 加载训练好的模型参数
        self.model.eval()  # 设置为评估模式

    def clear_canvas(self):
        self.draw_widget.clear()  # 清除画布

    def predict_digit(self):
        qt_image = self.draw_widget.get_image()  # 获取绘制的图像
        qt_image = qt_image.convertToFormat(QImage.Format_Grayscale8)  # 转换为灰度图像
        width = qt_image.width()
        height = qt_image.height()
        ptr = qt_image.bits()
        ptr.setsize(height * width)  # 设置指针大小
        arr = np.frombuffer(ptr, np.uint8).reshape((height, width))  # 将指针数据转换为numpy数组
        image = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)  # 转换为RGB图像
        image = cv2.resize(image, (224, 224))  # 调整图像大小
        image = image.astype(np.float32) / 255.0  # 归一化
        image = (image - 0.5) / 0.5  # 标准化
        image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)  # 转换为PyTorch张量

        with torch.no_grad():
            output = self.model(image)  # 预测
            _, predicted = torch.max(output.data, 1)  # 获取预测结果
            digit = predicted.item()
            self.label.setText(f"预测结果: {digit}")  # 显示预测结果

# 程序入口
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
