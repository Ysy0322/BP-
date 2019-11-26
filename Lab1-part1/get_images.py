#-*- coding: utf-8 -*-
import os
import numpy
from PIL import Image   #导入Image模块
from pylab import *     #导入savetxt模块

#读取图片信息。
#以下代码看可以读取文件夹下所有文件
# def getAllImages(folder):
#     assert os.path.exists(folder)
#     assert os.path.isdir(folder)
#     imageList = os.listdir(folder)
#     imageList = [os.path.abspath(item) for item in imageList if os.path.isfile(os.path.join(folder, item))]
#     return imageList

# print getAllImages(r"D:\\test")



def get_imlist(path):   #此函数读取特定文件夹下的bmp格式图像
    return [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.bmp')]

image_info = [0.0] * 14
def get_images_info():
    image_paths = [0.0] * 14
    for i in range(14):
        image_paths[i] = get_imlist("TRAIN\\"+str(i+1))

    for i in range(14):
        image_info[i] = [0.0] * len(image_paths[i])
        for j in range(len(image_paths[i])):
            image = Image.open(image_paths[i][j])
            img_ndarray = numpy.asarray(image)
            image_info[i][j] = numpy.ndarray.flatten(img_ndarray) #将图像的矩阵形式转化为一维矩阵
        #savetxt("data"+str(i+1)+".txt",image_info[i],fmt="%.0f")


def get_data():
    get_images_info()
    length = int(256 * 0.8) # 每一组是256个字，将每一组的80%用作训练集，另外20%用作测试机
    train_data = [0.0] * (length*14)
    train_label = [0.0] * (length*14)
    test_data = [0.0] * ((256-length)*14)
    test_label = [0.0] * ((256-length)*14)

    for i in range(length):
            for j in range(14):
                index = 14 * i + j
                train_data[index] =image_info[j][i]
                train_label[index] = j #get_label(j,14)
    for i in range(256-length):
        for j in range(14):
            index = 14 * i + j
            img_index = (length+i)
            test_data[index] = image_info[j][img_index]
            test_label[index] = j # get_label(j,14)
    return train_data, train_label, test_data, test_label

    '''for i in range(14):
        for j in range(length):
            index = length * i + j
            train_data[index] =image_info[i][j]
            train_label[index] = get_label(i,14)
    length2 = 256-length
    for i in range(14):
        for j in range(length2):
            index = (length2) * i + j
            img_index = (length+j)
            test_data[index] = image_info[i][img_index]
            test_label[index] = get_label(i,14)
    return train_data, train_label, test_data, test_label'''



def get_label(index,k):
    label = [0] * 14
    for i in range(k):
        if i == index:
            label[i] = 1
        else:
            label[i] = 0
    return label



if __name__ == '__main__':
    train_data, train_label, test_data, test_label = get_data()
    savetxt("train_data",train_data,fmt="%.0f")
    savetxt ("train_label", train_label, fmt="%.0f")
    savetxt ("test_data", test_data, fmt="%.0f")
    savetxt ("test_label", test_label, fmt="%.0f")