# -*- coding: utf-8 -*-
import os
import numpy as np
L = []


def listdir(path):
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            L.append(file)
    return L

if __name__ == '__main__':
    #listdir("/home/yuzhg/my-tf-faster-rcnn-simple/data/yzg1/3147")
    a =np.array(listdir("/home/yuzhg/my-tf-faster-rcnn-simple/data/yzg1/3147"))
