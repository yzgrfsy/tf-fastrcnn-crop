# -*- coding: utf-8 -*-
"""
Created on Sun Jun 11 18:04:24 2017

@author: wuhui
用来对图片进行重命名，并初始化目录结构
"""

import cv2
import os
from xml.etree.ElementTree import ElementTree
import  sys,os
import struct

def modifiedXml(xmlfile):
    tree = ElementTree()
    tree.parse(xmlfile)
    all_objects = tree.getroot().getchildren()

    for object in all_objects:
        if object.tag == "filename":
            object.text = os.path.splitext(os.path.split(xmlfile)[1])[0] + ".jpg"
            print object.text
        if object.tag == "object":
            object.find("name").text = "1"



    tree.write(xmlfile,encoding="utf-8")

def rename(path):
    xmls = os.listdir(path + "/Annotations")
    print xmls
    imgs = os.listdir(path + "/JPEGImages")
    print imgs
    cnt = 1
    prename = "000000"
    for img,xml in zip(imgs,xmls):
        print img,xml
        '''
        print path +"/JPEGImages/" + img
        print path + "/JPEGImages/" + prename[0:len(prename) - len(str(cnt))] + str(cnt) + ".jpg"
        os.rename(path +"/JPEGImages/" + img, path + "/JPEGImages/" + prename[0:len(prename) - len(str(cnt))] + str(cnt) + ".jpg")
        #modified the xml file
        newName = path + "/Annotations/" + prename[0:len(prename) - len(str(cnt))] + str(cnt) + ".xml"
        os.rename(path + "/Annotations/" + xml, newName)
        modifiedXml(newName)
        cnt += 1
        '''
    print cnt
    print 'done!'

if __name__ == '__main__':
    rename("/home/chenxingli/dengtaAI/my-tf-faster-rcnn/data/VOCdevkit2007/VOC2007")
    #modifiedXml("/home/chenxingli/dengtaAI/my-tf-faster-rcnn/data/VOCdevkit2007/VOC2007/test/Annotations/000001.xml")
