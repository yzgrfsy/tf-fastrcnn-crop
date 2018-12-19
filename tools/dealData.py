import os
import random

# train : val : test = 7:2:1
trainval_percent = 0.9
train_percent = 0.78
##############################################################################
'''
xmlfilepath = '/home/chenxingli/dengtaAI/my-tf-faster-rcnn/data/VOCdevkit2007/VOC2007/Annotations'
imagefilepath = '/home/chenxingli/dengtaAI/my-tf-faster-rcnn/data/VOCdevkit2007/VOC2007/JPEGImages'
txtsavepath = '/home/chenxingli/dengtaAI/my-tf-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main'
'''
pre_path = '/home/hdd300g/modelers/modeler001/my-tf-faster-rcnn-simple'
xmlfilepath = pre_path + '/data/VOCdevkit2007/VOC2007/Annotations'
imagefilepath = pre_path + '/data/VOCdevkit2007/VOC2007/JEPGImages'
txtsavepath = pre_path + '/data/VOCdevkit2007/VOC2007/ImageSets/Main' 
##############################################################################
total_xml = os.listdir(xmlfilepath) # total_xml ->  type:list ['000001.xml', ..., '009224.xml']

num=len(total_xml) # num : 9224
list=range(num) # list : [0, 1, ..., 9223]
tv=int(num*trainval_percent) # int(9224 * 0.9) = 8301 
tr=int(tv*train_percent) # int(0.78 * 8301) = 6474
trainval= random.sample(list,tv) # randomly get tv=8301 numbers from list[0,1,...,9223]
train=random.sample(trainval,tr) 

#####################################################################################
'''
ftrainval = open('/home/chenxingli/dengtaAI/my-tf-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt', 'w')
ftest = open('/home/chenxingli/dengtaAI/my-tf-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt', 'w')
ftrain = open('/home/chenxingli/dengtaAI/my-tf-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/train.txt', 'w')
fval = open('/home/chenxingli/dengtaAI/my-tf-faster-rcnn/data/VOCdevkit2007/VOC2007/ImageSets/Main/val.txt', 'w')
'''
ftrainval = open(txtsavepath + '/trainval.txt', 'w')
ftest = open(txtsavepath + '/test.txt', 'w')
ftrain = open(txtsavepath + '/train.txt', 'w')
fval = open(txtsavepath + 'val.txt', 'w')
######################################################################################
def deal():
    for i  in list:
        name=total_xml[i][:-4]+ '\n'  # such as : total_xml[i] is '000001.xml' and name is '000001\n'
        if i in trainval:
           ftrainval.write(name)
           if i in train:
              ftrain.write(name)
           else:
              fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest.close()

if __name__ == "__main__":
    deal()
