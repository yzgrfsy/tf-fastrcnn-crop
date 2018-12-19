# -*- coding:utf-8 -*-

import os
#rename image from 000000 by huanghao
class ImageRename():
    def __init__(self):
        self.path = '/home/yuzhg/my-tf-faster-rcnn-simple/data/2018-11-5 '

    def rename(self):
        filelist = os.listdir(self.path)
        total_num = len(filelist)

        i = 0

        for item in filelist:
            if item.endswith('.jpg'):
                src = os.path.join(os.path.abspath(self.path), item)
                dst = os.path.join(os.path.abspath(self.path), '0000' + format(str(i), '0>3s') + '.jpg')
                os.rename(src, dst)
                i = i + 1
        print (total_num, i)

if __name__ == '__main__':
    newname = ImageRename()
    newname.rename()