import os

def delete_xml_files(folder_path):
    try:
        # 获取文件夹中所有文件和子文件夹的列表
        files_and_folders = os.listdir(folder_path)

        # 遍历每个文件或子文件夹
        for item in files_and_folders:
            item_path = os.path.join(folder_path, item)

            # 如果是文件夹，递归调用 delete_xml_files
            if os.path.isdir(item_path):
                delete_xml_files(item_path)

            # 如果是.xml文件，删除
            elif item.lower().endswith('.xml'):
                os.remove(item_path)
                print(f"Deleted: {item_path}")

    except Exception as e:
        print(f"An error occurred: {e}")

def count_png_files(folder_path):
    try:
        # 获取文件夹中所有文件和子文件夹的列表
        files_and_folders = os.listdir(folder_path)

        # 初始化计数器
        png_count = 0

        # 遍历每个文件或子文件夹
        for item in files_and_folders:
            item_path = os.path.join(folder_path, item)

            # 如果是文件夹，递归调用 count_png_files
            if os.path.isdir(item_path):
                png_count += count_png_files(item_path)

            # 如果是.png文件，增加计数器
            elif item.lower().endswith('.png'):
                png_count += 1

        return png_count

    except Exception as e:
        print(f"An error occurred: {e}")
        return 0 

# 使用示例
folder_path = 'D:\\Infrared small object detection\\record\\code-set\\IEEE_TIP_UIU-Net\\masks'
total_png_files = count_png_files(folder_path)
print(f"Total number of .png files: {total_png_files}")
