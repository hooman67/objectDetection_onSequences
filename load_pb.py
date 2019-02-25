import tensorflow as tf
import numpy as np
import cv2
from utils import decode_netout, draw_boxes
import matplotlib.pyplot as plt
import os
from time import time

# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]="0"

obj_threshold = 0.3
nms_threshold = 0.01

def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the 
    # unserialized graph_def
    with tf.device('/device:GPU:0'):
        with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # Then, we import the graph_def into a new Graph and returns it 
        with tf.Graph().as_default() as graph:
            # The name var will prefix every op/nodes in your graph
            # Since we load everything in a new graph, this is not needed
            tf.import_graph_def(graph_def, name="prefix")
    return graph


def predict(sess, y, x, image, image_path):
    image_h, image_w, _ = image.shape
    image = cv2.resize(image, (640, 640))
    image = image / 255.
    # image = self.feature_extractor.normalize(image)

    input_image = image[:, :, ::-1]
    input_image = np.expand_dims(input_image, 0)
    input_image = np.expand_dims(input_image, 0)

    # netout = self.model.predict(input_image)[0]
    start_time = time()
    netout = sess.run(y, {x: input_image})[0]

    anchors = [0.31,0.81, 0.40,1.16, 0.41,0.91, 0.42,0.68, 0.50,1.42, 0.52,1.09, 0.63,1.72, 0.96,1.72, 1.14,2.69, 10.64,2.53]
    labels = ["tooth", "toothline"]
    nb_class = len(labels) 
    boxes  = decode_netout(netout, anchors, nb_class, obj_threshold, nms_threshold)
    print("Time spent: ", time() - start_time)

    image = draw_boxes(image, boxes, labels, obj_threshold)
    # plt.imshow(image)
    # plt.show()
    path_to_save = pb_filepath.split('/')[:-1]
    path_to_save = '/'.join(path_to_save)
    path_to_save = os.path.join(path_to_save, image_path.split('/')[-1])
    print(path_to_save, image.shape)
    image *= 255
    cv2.imwrite(path_to_save, image)

    return boxes


if __name__ == '__main__':
    # Let's allow the user to pass the filename as an argument
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--frozen_model_filename", default="results/frozen_model.pb", type=str, help="Frozen model file to import")
    # args = parser.parse_args()

    pb_filepath = "/home/cheng/Desktop/repos/keras-yolo2/tf_graph/conv_lstm.pb"
    # We use our "load_graph" function
    graph = load_graph(pb_filepath)
    print("Graph loaded...")

    # We can verify that we can access the list of operations in the graph
    for op in graph.get_operations():
        print(op.name) 
        # prefix/Placeholder/inputs_placeholder
        # ...
        # prefix/Accuracy/predictions
        
    # We access the input and output nodes 
    # x = graph.get_tensor_by_name('prefix/Placeholder/inputs_placeholder:0')
    # x = graph.get_tensor_by_name('prefix/input_1:0')
    x = graph.get_tensor_by_name('prefix/images_input_1:0')
    y = graph.get_tensor_by_name('prefix/reshape_3/Reshape:0')

    # image_path = "/home/cheng/Desktop/data/optical/teeth/Test/image/123_97882.jpg"
    # img = cv2.imread(image_path)
    # img = cv2.resize(img, (640, 640))
    # img = img[None, ...]
        
    input_path = "/home/cheng/Desktop/data/optical/teeth/Test/image/" 
    image_paths = []

    if os.path.isdir(input_path): 
        for inp_file in os.listdir(input_path):
            image_paths += [input_path + inp_file]
    else:
        image_paths += [input_path]

    image_paths = [inp_file for inp_file in image_paths if (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]
    # We launch a Session
    with tf.device('/device:GPU:0'):
        with tf.Session(graph=graph) as sess:
            # Note: we don't nee to initialize/restore anything
            # There is no Variables in this graph, only hardcoded constants 
            # y_out = sess.run(y, feed_dict={
            #     x: [[3, 5, 7, 4, 5, 1, 1, 1, 1, 1]] # < 45
            # })
            # y_out = sess.run(y, {x: img})

            # the main loop
            for image_path in image_paths:
                # print("Processing ", image_path)
                image = cv2.imread(image_path)
                if image is None:
                    image = cv2.imread(image_path)
                else:
                    boxes = predict(sess, y, x, image, image_path)
            # I taught a neural net to recognise when a sum of numbers is bigger than 45
            # it should return False in this case
            print(boxes) # [[ False ]] Yay, it works!

