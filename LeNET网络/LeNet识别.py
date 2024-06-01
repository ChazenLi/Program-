import tkinter as tk
from tkinter import filedialog
import numpy as np
from PIL import Image, ImageDraw
import torch
import torch.nn as nn
import torchvision.transforms as transforms

# 定义LeNet模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.dropout1 = nn.Dropout(p=0.7)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(p=0.7)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool1(torch.relu(self.conv1(x)))
        x = self.pool2(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 6 * 6)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

# 加载预训练模型
model = LeNet()
model.load_state_dict(torch.load('E:/Python-projects/pytorch/lenet.pth'))
model.eval()

# 定义图像变换
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# GUI应用
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("手写数字识别")
        self.canvas = tk.Canvas(self, width=200, height=200, bg='white')
        self.canvas.pack()
        
        self.button_clear = tk.Button(self, text='清除', command=self.clear_canvas)
        self.button_clear.pack()
        
        self.button_predict = tk.Button(self, text='识别', command=self.predict_digit)
        self.button_predict.pack()
        
        self.label_result = tk.Label(self, text='', font=('Helvetica', 20))
        self.label_result.pack()
        
        self.canvas.bind('<B1-Motion>', self.paint)
        self.image = Image.new('L', (200, 200), 'white')
        self.draw = ImageDraw.Draw(self.image)
        
    def clear_canvas(self):
        self.canvas.delete('all')
        self.image = Image.new('L', (200, 200), 'white')
        self.draw = ImageDraw.Draw(self.image)
        self.label_result.config(text='')
        
    def paint(self, event):
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        self.canvas.create_oval(x1, y1, x2, y2, fill='black', width=10)
        self.draw.line([x1, y1, x2, y2], fill='black', width=10)
        
    def predict_digit(self):
        img = self.image.resize((28, 28))
        img = img.resize((32, 32), Image.LANCZOS)  # 使用Image.LANCZOS代替Image.ANTIALIAS
        img = transform(img).unsqueeze(0)
        
        with torch.no_grad():
            output = model(img)
            _, predicted = torch.max(output, 1)
            self.label_result.config(text=f'识别结果: {predicted.item()}')

if __name__ == '__main__':
    app = App()
    app.mainloop()
