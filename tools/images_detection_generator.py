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

import argparse
import cv2
import os

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math
import _init_paths

from model.config import cfg
from model.nms_wrapper import nms
from model.test import im_detect
from nets.resnet_v1 import resnetv1
from nets.vgg16 import vgg16
from utils.timer import Timer

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

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_10000.ckpt',), 'res101': ('res101_faster_rcnn_iter_110000.ckpt',)}
DATASETS = {'pascal_voc': ('voc_2007_trainval',), 'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def save_to_file(file_name, contents):
    try:
        fh = open(file_name, 'w')
        fh.write(contents)
    except (IOError,TypeError) as x:
        print(x)
        print(contents)
        print(file_name)
    finally:
        fh.close()

def eachFile(filepath,xmlsavepath,sess,net):
    list = []
    if(os.path.isdir(filepath) == False):
        return list
    pathDir = os.listdir(filepath)
    pathDir.sort()
    for allDir in pathDir:
        child = os.path.join(filepath, allDir)
        strcontent = demo(sess, net, child)
        save_to_file(os.path.join(xmlsavepath, allDir.split(".")[0]+".xml"), strcontent)

def create_xml(width, height, bbox_list):
    template = '''
<annotation>
    <folder>folder</folder>
    <filename>filename.jpg</filename>
    <size>
        <width>{width}</width>
        <height>{height}</height>
        <depth>3</depth>
    </size>
    {bbox}
</annotation>'''
    object_template = '''
<object>
<name>{name}</name>
<difficult>0</difficult>
    <bndbox>
        <xmin>{xmin}</xmin>
        <ymin>{ymin}</ymin>
        <xmax>{xmax}</xmax>
        <ymax>{ymax}</ymax>
    </bndbox>
</object>
'''
    template = template.format(width=width, height=height, bbox='{bbox}')
    bbox = []
    bbox_list = sorted(bbox_list, key=lambda bnb: bnb[0], reverse=False)
    for b in bbox_list:
        bbox.append(object_template.format(xmin=int(round(b[0])),
                                           ymin=int(round(b[1])),
                                           xmax=int(round(b[2])),
                                           ymax=int(round(b[3])),
                                           name='text'))
    return template.format(bbox=''.join(bbox))


def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    mask = im
    im = im[:, :, (2, 1, 0)]
    # print(im.shape())
    fig, ax = plt.subplots(figsize=(12, 12))
    # ax.imshow(im, aspect='equal')
    # mask = im#.transpose((1, 2, 0)).astype(np.uint8).copy()  #np.zeros_like(im, dtype=np.uint8).copy()
    bbox_account = 0
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        bbox_account += 1
        # '''
        cv2.rectangle(mask, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 1)
        font = cv2.FONT_HERSHEY_SIMPLEX
        text = '{:s}\{:.3f}'.format(class_name, score)
        cv2.putText(mask, text, (int(bbox[0]), int(bbox[1] - 2)), font, 1, (0, 0, 255), 2)
        # '''

        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
        ax.text(bbox[0], bbox[1] - 2,
                '{:s} {:.3f}'.format(class_name, score),
                bbox=dict(facecolor='blue', alpha=0.5),
                fontsize=14, color='white')

    ax.set_title(('{} detections with '
                  'p({} | box) >= {:.1f}').format(class_name, class_name,
                                                  thresh),
                 fontsize=14)
    plt.axis('off')
    plt.tight_layout()
    plt.draw()

    print("------------------------------------------------------------")
    print("the account of bbox is:", bbox_account)
    return mask


def demo(sess, net, image_path_name):
    """Detect object classes in an image using pre-computed object proposals."""

    # Load the demo image
    # im_file = os.path.join(cfg.DATA_DIR, 'demo', image_name)
    im = cv2.imread(image_path_name)

    # Detect all object classes and regress object bounds
    timer = Timer()
    timer.tic()
    scores, boxes = im_detect(sess, net, im)
    timer.toc()
    #print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, boxes.shape[0]))
    print('Detection for images : '+image_path_name)
    # Visualize detections for each class
    CONF_THRESH = 0.8
    NMS_THRESH = 0.3

    box_list = []
    for cls_ind, cls in enumerate(CLASSES[1:]):
        cls_ind += 1  # because we skipped background
        cls_boxes = boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
        cls_scores = scores[:, cls_ind]
        dets = np.hstack((cls_boxes,
                          cls_scores[:, np.newaxis])).astype(np.float32)
        keep = nms(dets, NMS_THRESH)
        dets = dets[keep, :]
        inds = np.where(dets[:, -1] >= CONF_THRESH)[0]
        if len(inds) == 0:
            break
        #mask = im
        im = im[:, :, (2, 1, 0)]
        # print(im.shape())
        #fig, ax = plt.subplots(figsize=(12, 12))
        # ax.imshow(im, aspect='equal')
        # mask = im#.transpose((1, 2, 0)).astype(np.uint8).copy()  #np.zeros_like(im, dtype=np.uint8).copy()
        for i in inds:
            bbox = dets[i, :4]
            #score = dets[i, -1]
            box_list.append(bbox)
    height, width = im.shape[:2]
    file_content = create_xml(width, height, box_list)

    return file_content

        # No need to visualize the image.
        # img = vis_detections(im, cls, dets, thresh=CONF_THRESH)
    #return img


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser(description='Tensorflow Faster R-CNN demo')
    parser.add_argument('--net', dest='demo_net', help='Network to use [vgg16 res101]',
                        choices=NETS.keys(), default='res101')
    parser.add_argument('--dataset', dest='dataset', help='Trained dataset [pascal_voc pascal_voc_0712]',
                        choices=DATASETS.keys(), default='pascal_voc_0712')
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    cfg.TEST.HAS_RPN = True  # Use RPN for proposals
    args = parse_args()

    # model path
    demonet = args.demo_net
    print("demonet is %s" % demonet)
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                           NETS[demonet][0])
    print("tfmodel is %s" % tfmodel)

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

    images_path = "/home/hdd300g/modelers/modeler001/hequn-imagexml-predict/images0304"
    xmls_path = "/home/hdd300g/modelers/modeler001/hequn-imagexml-predict/xmls0304"
    paths = os.listdir(images_path)
    paths.sort()
    for filename in paths:
        print(filename)
        eachFile(os.path.join(images_path, filename), xmls_path, sess, net)
    images_path = "/home/hdd300g/modelers/modeler001/hequn-imagexml-predict/images0506"
    xmls_path = "/home/hdd300g/modelers/modeler001/hequn-imagexml-predict/xmls0506"
    paths = os.listdir(images_path)
    paths.sort()
    for filename in paths:
        print(filename)
        eachFile(os.path.join(images_path, filename), xmls_path, sess, net)
