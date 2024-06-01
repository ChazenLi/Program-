import os

train_data_path = 'E:/Python-projects/pytorch/练习/项目实战/mnist_images/train'
test_data_path = 'E:/Python-projects/pytorch/练习/项目实战/mnist_images/test'

# 检查训练数据路径
if not os.path.exists(train_data_path):
    print(f"Train data path {train_data_path} does not exist.")
else:
    print(f"Train data path {train_data_path} exists.")

if not os.path.exists(test_data_path):
    print(f"Test data path {test_data_path} does not exist.")
else:
    print(f"Test data path {test_data_path} exists.")

# 确认子文件夹和文件存在
for i in range(10):
    folder_path = os.path.join(train_data_path, str(i))
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
    else:
        files = os.listdir(folder_path)
        if not files:
            print(f"No files found in folder {folder_path}.")
        else:
            print(f"Found {len(files)} files in folder {folder_path}.")
            for file in files[:5]:  # 只打印前5个文件
                print(f"File: {file}")

# 类似地检查测试数据路径
for i in range(10):
    folder_path = os.path.join(test_data_path, str(i))
    if not os.path.exists(folder_path):
        print(f"Folder {folder_path} does not exist.")
    else:
        files = os.listdir(folder_path)
        if not files:
            print(f"No files found in folder {folder_path}.")
        else:
            print(f"Found {len(files)} files in folder {folder_path}.")
            for file in files[:5]:  # 只打印前5个文件
                print(f"File: {file}")
