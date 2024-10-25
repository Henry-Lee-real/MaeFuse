import os

def name():
    current_directory = os.getcwd()  # 获取当前目录的路径
    parent_directory = os.path.dirname(current_directory)  # 获取上一级目录的路径

    print("当前目录:", current_directory)
    print("上一级目录:", parent_directory)
