import os
import random
import shutil

# 设置路径
train_input_path = r'E:\PHD\UIEB\base\UIEB\train\input'
train_target_path = r'E:\PHD\UIEB\base\UIEB\train\target'
test_input_path = r'E:\PHD\UIEB\base\UIEB\test\input'
test_target_path = r'E:\PHD\UIEB\base\UIEB\test\target'

# 获取所有图片文件名
image_files = os.listdir(train_input_path)
random.shuffle(image_files)

# 从train中选取90张图片
random_images = image_files[:90]

# 移动图片和对应的target文件
for image in random_images:
    image_number = image.split('.')[0]  # 获取图片序号
    target_file = f"{image_number}.png"  # 对应的target文件名

    # 移动图片
    shutil.move(os.path.join(train_input_path, image), os.path.join(test_input_path, image))

    # 移动target文件
    shutil.move(os.path.join(train_target_path, target_file), os.path.join(test_target_path, target_file))