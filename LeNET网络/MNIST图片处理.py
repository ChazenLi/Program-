import os
import gzip
import numpy as np
from PIL import Image

# MNIST 数据集文件路径
mnist_path = 'E:\\training data\\MNIST_data\\MNIST_data'

# 创建保存图片的文件夹
def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# 读取 MNIST 数据集文件
def read_mnist_images(filename):
    with gzip.open(filename, 'rb') as f:
        data = f.read()
    images = np.frombuffer(data, np.uint8, offset=16)
    images = images.reshape(-1, 28, 28)
    return images

def read_mnist_labels(filename):
    with gzip.open(filename, 'rb') as f:
        data = f.read()
    labels = np.frombuffer(data, np.uint8, offset=8)
    return labels

# 保存图片
def save_images(images, labels, directory):
    for i, (image, label) in enumerate(zip(images, labels)):
        label_dir = os.path.join(directory, str(label))
        create_dir(label_dir)
        image = Image.fromarray(image, 'L')
        image.save(os.path.join(label_dir, f'{i}.png'))

# 文件名
train_images_file = os.path.join(mnist_path, 'train-images-idx3-ubyte.gz')
train_labels_file = os.path.join(mnist_path, 'train-labels-idx1-ubyte.gz')
test_images_file = os.path.join(mnist_path, 't10k-images-idx3-ubyte.gz')
test_labels_file = os.path.join(mnist_path, 't10k-labels-idx1-ubyte.gz')

# 读取数据
train_images = read_mnist_images(train_images_file)
train_labels = read_mnist_labels(train_labels_file)
test_images = read_mnist_images(test_images_file)
test_labels = read_mnist_labels(test_labels_file)

# 保存训练集图片
train_dir = 'mnist_images/train'
create_dir(train_dir)
save_images(train_images, train_labels, train_dir)

# 保存测试集图片
test_dir = 'mnist_images/test'
create_dir(test_dir)
save_images(test_images, test_labels, test_dir)

print('Images have been saved successfully.')
