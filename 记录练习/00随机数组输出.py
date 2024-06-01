import random

# 生成随机数组长度
array_length = 10

# 生成随机浮点数数组并保留两位小数
random_float_array = [round(random.uniform(0, 20), 2) for _ in range(array_length)]

# 对数组进行排序（原地排序）
random_float_array.sort()

# 输出数组格式
print(random_float_array)

