import tensorflow as tf
import os

graph_def_file = "./frozen_model.pb"
input_arrays = ["Placeholder_1"]
output_arrays = ["MobilenetV1_4/cls_prob","MobilenetV1_4/bbox_pred/BiasAdd",
                 "MobilenetV1_2/rois/concat","MobilenetV1_4/cls_score/BiasAdd" ]

converter = tf.contrib.lite.TocoConverter.from_frozen_graph(
  graph_def_file, input_arrays, output_arrays)
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)

