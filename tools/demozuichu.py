#!/usr/bin/env python

# --------------------------------------------------------
# Tensorflow Faster R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Xinlei Chen, based on code from Ross Girshick
# --------------------------------------------------------

"""
Demo script showing detections in sample images.

See README.md for installation instructions before running.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from PIL import Image
import _init_paths
from model.config import cfg
from model.test import im_detect
from model.nms_wrapper import nms
from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

from nets.vgg16 import vgg16
from nets.resnet_v1 import resnetv1

'''
CLASSES = ('__background__',
           'aeroplane', 'bicycle', 'bird', 'boat',
           'bottle', 'bus', 'car', 'cat', 'chair',
           'cow', 'diningtable', 'dog', 'horse',
           'motorbike', 'person', 'pottedplant',
           'sheep', 'sofa', 'train', 'tvmonitor')
'''
CLASSES = ('__background__',
           '1')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_10000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    # print('dets shape:', dets.shape)
    # print('dets[:, -1]:', dets[:, -1])
    # print('index:', np.where(dets[:, -1] >= thresh))
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    #print('len(inds):', len(inds))
    mask = im
    im = im[:, :, (2, 1, 0)]
    bbox_account = 0
    #m = 1
    for i in inds:
        bbox = dets[i, :4]
        # print('...........................')
        # print(bbox)
        score = dets[i, -1]
        bbox_account = bbox_account + 1
        cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        text = '{:s}\{:.3f}'.format(class_name, score)
        cv2.putText(mask, text, (int(bbox[0]), int(bbox[1] - 2)), font, 2, (0, 0, 255), 1)

        #img_crop = im[int(bbox[1]):int(bbox[3]), int(bbox[0]): int(bbox[2])]
        #cv2.imwrite("/home/yuzhg/my-tf-faster-rcnn-simple/data/imagecrop/"  + str(m) + "_crop" + ".jpg", img_crop)
        #cv2.imwrite("/home/yuzhg/my-tf-faster-rcnn-simple/data/imagecrop/"+str(m)+".jpg", img_crop)
        #m = m + 1
    print("the nubmer of object is:", bbox_account)
    return mask


def crop(im, dets, thresh=0.5):
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    print('len(inds):', len(inds))
    mask = im
    #im = im[:, :, (2, 1, 0)]  # nose become blue

    m = 1
    while m <= len(inds):
        for i in inds:
            bbox = dets[i, :4]
            img_crop = im[int(bbox[1])+2:int(bbox[3]-1), int(bbox[0])+2: int(bbox[2])-1]
            cv2.imwrite("/home/yuzhg/my-tf-faster-rcnn-simple/data/imagecrop/" + im_name[0:7] + "_" + str(
                m) + "_" + "crop_result" + ".jpg", img_crop)
            m += 1


def demo(sess, net, image_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(im_file)
    # print("im.shape", im.shape)
    # print(im.shape[0])
    # print(im.shape[1])
    # print(im.shape[2])
    #print('im:', im)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))

    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4*cls_ind:4*(cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        #print('dets:', dets)
        #print('im is None:', im)
        img = vis_detections(im, cls, dets, thresh=CONF_THRESH)
        crop(im, dets, thresh=0.5)

    return img


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args


L = []


def listdir(path):
    for file in os.listdir(path):
        if file.endswith('.jpg'):
            L.append(file)
    return L




if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    print("demonet is %s" %demonet)
    dataset = args.dataset
    tfmodel = os.path.join('/home/yuzhg/my-tf-faster-rcnn-simple/output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])
    print("tfmodel is %s"%tfmodel)


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '  
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True

    # init session
    sess = tf.Session(config=tfconfig)
    # load network
    if demonet == 'vgg16':
        net = vgg16()
    elif demonet == 'res101':
        net = resnetv1(num_layers=101)
    else:
        raise NotImplementedError
    net.create_architecture("TEST", 2,
                          tag='default', anchor_scales=[8, 16, 32])
    saver = tf.train.Saver()
    saver.restore(sess, tfmodel)

    print('Loaded network {:s}'.format(tfmodel))
    im_names = np.array(listdir("/home/yuzhg/my-tf-faster-rcnn-simple/data/demo/"))
    print(im_names)
    for im_name in im_names:
        # print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        # print('Demo for data/demo/{}'.format(im_name))
        print(im_name)
        # print(im_name[0:7])
        im = demo(sess, net, im_name)
        cv2.imwrite("/home/yuzhg/my-tf-faster-rcnn-simple/data/detect_result/" + im_name[0:7] + "detect_result" + ".jpg", im)
       # cv2.imwrite("/home/yuzhg/my-tf-faster-rcnn-simple/data/imagecrop/" + im_name[0:7] +"_"+ str(m) +"_"+"crop_result" +".jpg", img1)

    #plt.show()


