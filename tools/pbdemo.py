from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import _init_paths
from model.config import cfg

from model.test import im_detect,_get_blobs, _get_image_blob,_clip_boxes,_rescale_boxes,apply_nms,test_net
from model.test import *
#from model.nms_wrapper import nms
from tensorflow.python.platform import gfile
from utils.timer import Timer
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import os, cv2
import argparse

#from nets.vgg16 import vgg16
#from nets.resnet_v1 import resnetv1


CLASSES = ('__background__',
           '1')

NETS = {'vgg16': ('vgg16_faster_rcnn_iter_60000.ckpt',),'res101': ('res101_faster_rcnn_iter_110000.ckpt',),'mobile': ('mobile_faster_rcnn_iter_60000.ckpt',)}
DATASETS= {'pascal_voc': ('voc_2007_trainval',),'pascal_voc_0712': ('voc_2007_trainval+voc_2012_trainval',)}

def vis_detections(im, class_name, dets, thresh=0.5):
    """Draw detected bounding boxes."""
    #print(class_name)
    inds = np.where(dets[:, -1] >= thresh)[0]
    if len(inds) == 0:
        return
    bbox_account = 0
    for i in inds:
        bbox = dets[i, :4]
        score = dets[i, -1]
        bbox_account=bbox_account+1
        cv2.rectangle(im, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 4)
        font = cv2.FONT_HERSHEY_COMPLEX_SMALL
        text = '{:s}\{:.3f}'.format(class_name, score)
        cv2.putText(im, text, (int(bbox[0]), int(bbox[1] - 2)), font, 2, (0, 0, 255), 1)
        print(text+","+(str(bbox[0]))+","+str(bbox[1])+","+str(bbox[2])+","+str(bbox[3]))
    print("the nubmer of object is:",bbox_account)

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
    print("demonet is %s" %demonet)
    dataset = args.dataset
    tfmodel = os.path.join('output', demonet, DATASETS[dataset][0], 'default',
                              NETS[demonet][0])
    print("tfmodel is %s"%tfmodel)


    if not os.path.isfile(tfmodel + '.meta'):
        raise IOError(('{:s} not found.\nDid you download the proper networks from '
                       'our server and place them properly?').format(tfmodel + '.meta'))

    # set config
    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth=True
    pb_model_path = "./frozen_model.pb"
    sess = tf.Session(config=tfconfig)
    with gfile.FastGFile(pb_model_path,"rb") as f:
        output_graph_def = tf.GraphDef()
        output_graph_def.ParseFromString(f.read())
        sess.graph.as_default()
        #print(output_graph_def)
        tf.import_graph_def(output_graph_def,name="")
        #variable_name = [v.name for v in tf.all_variables()]
        f = open("./tensor_restore.txt", "w+")
        print(output_graph_def,file=f)

         # init session
    sess.run(tf.global_variables_initializer())

    image = sess.graph.get_tensor_by_name("Placeholder:0")
    image_info = sess.graph.get_tensor_by_name("Placeholder_1:0")
    scores_predict = sess.graph.get_tensor_by_name("MobilenetV1_4/cls_prob:0")
    boxes_predict = sess.graph.get_tensor_by_name("MobilenetV1_4/bbox_pred/BiasAdd:0")
    rois_predict = sess.graph.get_tensor_by_name("MobilenetV1_2/rois/concat:0")
    cls_score_predict = sess.graph.get_tensor_by_name("MobilenetV1_4/cls_score/BiasAdd:0")

    #cls_score_predict = sess.graph.get_tensor_by_name("vgg_16_3/cls_score/BiasAdd:0")
    im_names = [#"/home/chenxingli/dengtaAI/dataset/testimages/input/443048211133329708.jpg",
                #"/home/chenxingli/dengtaAI/dataset/testimages/input/119286569232544007.jpg",
                #"/home/chenxingli/dengtaAI/dataset/testimages/input/721687407986871341.jpg",
                "/home/hdd300g/modelers/modeler001/share/my-tf-faster-rcnn-simple/data/test/images/input/000100.jpg"
                ]
    for im_name in im_names:
        print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        print('Demo for data/demo/{}'.format(im_name))
        # Load the demo image
        im_file = os.path.join(cfg.DATA_DIR, 'demo', im_name)
        im = cv2.imread(im_file)

        # Detect all object classes and regress object bounds
        timer = Timer()
        timer.tic()
        blobs, im_scales = _get_blobs(im)
        assert len(im_scales) == 1, "Only single-image batch implemented"

        im_blob = blobs['data']
        blobs['im_info'] = np.array([im_blob.shape[1], im_blob.shape[2], im_scales[0]], dtype=np.float32)

        gt_boxes = np.zeros((1, 5), dtype=np.float32)
        # gt_boxes[:, 0:4] = roidb[0]['boxes'][gt_inds, :] * im_scales[0]
        # gt_boxes[:, 4] = roidb[0]['gt_classes'][gt_inds]
        blobs['gt_boxes'] = gt_boxes
        feed_dict = {image: blobs["data"],
                     image_info: blobs["im_info"],
                     #image_gt: blobs["gt_boxes"]
                     }

        _, scores, bbox_pred, rois= sess.run([cls_score_predict,
                                              scores_predict,
                                               boxes_predict,
                                               rois_predict
                                             ],
                                             feed_dict=feed_dict)
        stds = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_STDS), (2))
        means = np.tile(np.array(cfg.TRAIN.BBOX_NORMALIZE_MEANS), (2))
        '''
        mul/y
        mul
        '''
        bbox_pred *= stds
        bbox_pred += means  #
        f = open("./restoreModel.txt", "a+")
        print(scores,file=f)
        print("\n",file=f)
        print(bbox_pred,file=f)
        print("\n",file=f)

        print(rois, file=f)
        f.close()
        boxes = rois[:, 1:5] / im_scales[0]
        scores = np.reshape(scores, [scores.shape[0], -1])
        bbox_pred = np.reshape(bbox_pred, [bbox_pred.shape[0], -1])
        if cfg.TEST.BBOX_REG:
            # Apply bounding-box regression deltas
            box_deltas = bbox_pred
            pred_boxes = bbox_transform_inv(boxes, box_deltas)
            pred_boxes = _clip_boxes(pred_boxes, im.shape)
        else:
            # Simply repeat the boxes, once for each class
            pred_boxes = np.tile(boxes, (1, scores.shape[1]))

        timer.toc()
        # scores, boxes
        print('Detection took {:.3f}s for {:d} object proposals'.format(timer.total_time, pred_boxes.shape[0]))

        # Visualize detections for each class
        CONF_THRESH = 0.8
        NMS_THRESH = 0.3
        for cls_ind, cls in enumerate(CLASSES[1:]):
            cls_ind += 1  # because we skipped background
            cls_boxes = pred_boxes[:, 4 * cls_ind:4 * (cls_ind + 1)]
            cls_scores = scores[:, cls_ind]
            dets = np.hstack((cls_boxes,
                              cls_scores[:, np.newaxis])).astype(np.float32)
            keep = nms(dets, NMS_THRESH)
            dets = dets[keep, :]
            vis_detections(im, cls, dets, thresh=CONF_THRESH)
        cv2.imwrite("/home/hdd300g/modelers/modeler001/share/my-tf-faster-rcnn-simple/data/test/images/output/000100test.jpg", im)

