import os
import random

pre_path='/home/hdd300g/modelers/modeler001/my-tf-faster-rcnn-simple'
trainval_percent = 0.66 # 
train_percent = 0.66 #
xmlfilepath = pre_path+'/data/VOCdevkit2007/VOC2007/Annotations'
imagefilepath = pre_path+'/data/VOCdevkit2007/VOC2007/JPEGImages' # 
txtsavepath = pre_path+'/data/VOCdevkit2007/VOC2007/ImageSets/Main' # 
total_xml = sorted(os.listdir(xmlfilepath), key=lambda bnb: bnb, reverse=False)

num=len(total_xml)
list=range(num)
tv=int(num*trainval_percent) # 
tr=int(tv*train_percent) # 
trainval= random.sample(list,tv) # 
train=random.sample(trainval,tr) # 

ftrainval = open(pre_path+'/data/VOCdevkit2007/VOC2007/ImageSets/Main/trainval.txt', 'w')
ftest = open(pre_path+'/data/VOCdevkit2007/VOC2007/ImageSets/Main/test.txt', 'w')
ftrain = open(pre_path+'/data/VOCdevkit2007/VOC2007/ImageSets/Main/train.txt', 'w')
fval = open(pre_path+'/data/VOCdevkit2007/VOC2007/ImageSets/Main/val.txt', 'w')

def deal():
    for i in list:
        name=total_xml[i][:-4]+ '\n'
        j = (i % 12) % 3
        if j != 0: #3:2
           ftrainval.write(name)
           k = (i % 12)
           if k == 1 or k == 4 or k==7 or k==8 or k==10 or k==11:
              ftrain.write(name)
           else:
              fval.write(name)
        else:
            ftest.write(name)

    ftrainval.close()
    ftrain.close()
    fval.close()
    ftest .close()

if __name__ == "__main__":
    deal()
