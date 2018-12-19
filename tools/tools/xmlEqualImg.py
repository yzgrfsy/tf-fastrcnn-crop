
# -*- coding:utf-8 -*-
import os
import time
import random
import numpy
xmlfilepath = '/home/chenxingli/dengtaAI/my-tf-faster-rcnn/data/VOCdevkit2007/VOC2007/test/Annotations'
imagefilepath = '/home/chenxingli/dengtaAI/my-tf-faster-rcnn/data/VOCdevkit2007/VOC2007/test/JPEGImages'
total_xml = os.listdir(xmlfilepath)

total_image = os.listdir(imagefilepath)

num=len(total_xml)
image_num = len(total_image)
list=range(num)
list_image=range(image_num)

xml=[]
image=[]
def deal():
    for i  in list:
        name=total_xml[i][:-4]
        xml.append(name)
    for j in list_image:
        name=total_image[j][:-4]
        image.append(name)


def equal():
    intersection = [x for x in xml if x in set(image)]
    print(len(intersection))
    for i in total_xml:
        filePath = os.path.split(i)  # 分割出目录与文件
        fileMsg = os.path.splitext(filePath[1])  # 分割出文件与文件扩展名
        #print(filePath[1])
        fileExt = fileMsg[1]  # 取出后缀名(列表切片操作)
        fileName = fileMsg[0]
        if fileName not in intersection:
            os.remove(os.path.join(xmlfilepath,filePath[1]))

    for j in total_image:
        filePath = os.path.split(j)  # 分割出目录与文件
        fileMsg = os.path.splitext(filePath[1])  # 分割出文件与文件扩展名
        fileExt = fileMsg[1]  # 取出后缀名(列表切片操作)
        fileName = fileMsg[0]
        if fileName not in intersection:
            os.remove(os.path.join(imagefilepath,filePath[1]))

if __name__ == "__main__":
    deal()
    equal()