import os
import pickle
from PIL import Image

# CIFAR-10测试集文件路径
test_batch_file = '/home/hujie/datasets/cifar10/test_batch'

# 保存图像的目标文件夹
output_folder = "cifar10_test_images"
os.makedirs(output_folder, exist_ok=True)

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def save_cifar_images(data, labels, output_folder):
    for i in range(len(data)):
        image_data = data[i]
        label = labels[i]

        # CIFAR-10图像的尺寸为32x32x3
        image_data = image_data.reshape(3, 32, 32).transpose(1, 2, 0)
        
        # 创建PIL图像对象
        image = Image.fromarray(image_data)
        
        # 保存图像文件，使用标签作为文件名
        image_filename = os.path.join(output_folder, f"{label}_{i}.png")
        image.save(image_filename)

if __name__ == '__main__':
    # 加载测试集数据
    test_data_dict = unpickle(test_batch_file)
    
    # 解析图像数据和标签
    test_data = test_data_dict[b'data']
    test_labels = test_data_dict[b'labels']
    
    # 保存测试集图像到目标文件夹
    save_cifar_images(test_data, test_labels, output_folder)
    
    print(f"CIFAR-10测试集中的图像已保存到目标文件夹: {output_folder}")
