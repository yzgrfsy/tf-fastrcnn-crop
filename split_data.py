import os
import random
import shutil
'''
data/
     0001/
        001.jpg
        002.jpg
            ...
     0002/
        001.jpg
        002.jpg
            ...
change to
'''
'''
data/
    train/
        0001/
            001.jpg
            002.jpg
            ...
        0002/
            001.jpg
            002.jpg
            ...
    valid/
        0001/
            001.jpg
            002.jpg
            ...
        0002/
            001.jpg
            002.jpg
            ...
    test/
        0001/
            001.jpg
            002.jpg
            ...
        0002/
            001.jpg
            002.jpg
            ...     

'''


train_val_percent = 0.7
valid_percent = 0.3
#image_dir is the root dir of images
'''
data/
    0001/
       001.jpg
    0002/
       001.jpg
'''
images_dir = "/home/yuzhg/keras-transfer-learning-for-oxford102/data/"

save_path = os.path.abspath(".") +"/sorted/"
print(save_path)

num_class = len(os.listdir(images_dir))

def split_one_class(class_dir,catogray):
    print("class_dir:")
    print(class_dir)
    train_path = save_path + "train/" +  catogray
    if not os.path.exists(train_path):
        os.makedirs(train_path)
    test_path = save_path + "test/" + catogray
    if not os.path.exists(test_path):
        os.makedirs(test_path)
    valid_path = save_path + "valid/" + catogray
    if not os.path.exists(valid_path):
        os.makedirs(valid_path)
    all_images = os.listdir(class_dir)
    #print(all_images)
    num = len(all_images)
    #print("num:", num)
    train_val = int(num * train_val_percent)
    valid = int(train_val * valid_percent)
    list_all = range(num)# [0,1,...,num-1]
    #print("list_all", list_all)
    trainval_num = random.sample(list_all, train_val)
    print("trainval_num", trainval_num)
    print(len(trainval_num))
    #test_num = [i for i in list_all if i not in trainval_num]
    valid_num = random.sample(trainval_num, valid)
    print(len(valid_num))
    print("valid_num", valid_num)
    for i in list_all:
        if i in trainval_num:#move this image to this dir
           if i in valid_num:
               print(i)
               print("va")
               shutil.move(class_dir + "/"+all_images[i], valid_path+"/"+all_images[i])
               #move to valid dir
           else:
               print(i)
               print("tval")
               shutil.move(class_dir+"/"+all_images[i], train_path+"/"+all_images[i])
               #move to train dir

        else:
            print(i)
            print("00000000000000000")
            shutil.move(class_dir+"/"+ all_images[i],test_path+"/"+all_images[i])
            #move to test dir


def split_all_class(data_dir):
    just_one = False
    if os.path.isdir(data_dir):
        for i in os.listdir(data_dir):
            print(i)
            if os.path.isdir(data_dir+i):
               split_one_class(data_dir+i,i)
            else:
                just_one = True
                break
    if just_one:
        split_one_class(data_dir)


if __name__ == "__main__":
    split_all_class(images_dir)