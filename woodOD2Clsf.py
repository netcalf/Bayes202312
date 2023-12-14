import os
from PIL import Image

# 类别映射字典
category_mapping = {
    0: '0GB',
    1: '1HJ',
    2: '2LF',
    3: '3QK',
    4: '4SJ',
    5: '5SP',
    6: '6KD'
}
def crop_and_save(image_path, description_path):
    # 打开图片
    img = Image.open(image_path)
    img_width, img_height = img.size

    # 读取描述文件并裁剪图像
    with open(description_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            values = line.strip().split(' ')
            category = int(values[0])  # 类别
            center_x, center_y = float(values[1]) * img_width, float(values[2]) * img_height  # 中心坐标
            width, height = float(values[3]) * img_width, float(values[4]) * img_height  # 宽度和高度

            # 计算裁剪区域的左上角和右下角坐标
            left = int(center_x - width / 2)
            upper = int(center_y - height / 2)
            right = int(center_x + width / 2)
            lower = int(center_y + height / 2)

            # 检查裁剪区域是否超出图像边界，并调整裁剪区域
            left = max(left, 0)
            upper = max(upper, 0)
            right = min(right, img_width)
            lower = min(lower, img_height)

            # 检查是否有有效的裁剪区域
            if right > left and lower > upper:
                # 裁剪图像
                cropped_img = img.crop((left, upper, right, lower))

                # 创建文件夹保存裁剪后的图像
                folder_path = f"./{category_mapping.get(category, 'unknown')}/"
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                filename = os.path.splitext(os.path.basename(image_path))[0]
                cropped_img.save(f"{folder_path}{filename}_cropped_{left}_{upper}_{right}_{lower}.png")


def process_images(images_folder, labels_folder):
    # 遍历images文件夹中的图片文件
    for root, dirs, files in os.walk(images_folder):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png')):  # 确保是图片文件
                image_path = os.path.join(root, file)
                filename_without_extension = os.path.splitext(file)[0]
                label_file = os.path.join(labels_folder, f"{filename_without_extension}.txt")

                # 检查对应的描述文件是否存在
                if os.path.exists(label_file):
                    crop_and_save(image_path, label_file)


def count_and_save_statistics(images_folder):
    # 统计各个分类下的文件数量
    statistics = {}
    for category_name in category_mapping.values():
        folder_path = f"./{category_name}/"
        if os.path.exists(folder_path):
            file_count = len(
                [name for name in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, name))])
            statistics[category_name] = file_count

    # 打印统计结果
    print("分类文件数量统计：")
    for category_name, count in statistics.items():
        print(f"{category_name}: {count} 个文件")

    # 将统计结果保存到文本文件
    with open(os.path.join(images_folder, "statistics.txt"), 'w') as stats_file:
        stats_file.write("分类文件数量统计：\n")
        for category_name, count in statistics.items():
            stats_file.write(f"{category_name}: {count} 个文件\n")

if __name__ == "__main__":
    images_folder = "images"  # 图片文件夹路径
    labels_folder = "labels"  # 描述文件夹路径
    process_images(images_folder, labels_folder)
# 统计文件数量并保存统计结果
    count_and_save_statistics("./")