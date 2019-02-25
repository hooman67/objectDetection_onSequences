import argparse
import json
import os
from random import shuffle
from time import time
import pprint as pp

from tracker import TrackerModel
from freeze_session import freeze_session
from keras.models import load_model
import numpy as np
import tensorflow as tf
import cv2
from tqdm import tqdm
from utils import draw_boxes

obj_threshold = 0.2
is_save_tf_model = True
is_filter_bboxes = True
shovel_type = "Hydraulic"
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

def main(args):
    config_path = args.conf
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())
    pp.pprint(config)
    config['train']['batch_size'] = 1

    tracker_model = TrackerModel(config, is_inference=True, is_inference_ensemble=False)
    pp.pprint(vars(tracker_model))
    yolo_weights_path = "/home/cheng/Desktop/repos/keras-yolo2/snapshots/latest_all_shovel_full_yolo/full_yolo_34.h5"
    # tracker_model.full_model.load_weights(yolo_weights_path, by_name=True)
    weights_path = "/media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/yolo_v2_models/RNN/256_1x1_lstm_raw_images_2_sec_augment_after_activation_lr_reduced/ConvLSTM2D-Tracker-24.h5"
    weights_path = "/media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/yolo_v2_models/RNN/ConvLSTM2D-Tracker-44.h5"
    # tracker_model.detector.load_weights(yolo_weights_path)
    tracker_model.full_model.load_weights(weights_path)
    if args.input.endswith('.avi') or args.input.endswith('.mp4'):
        tracker_model.set_inference_weights()
        tracker_model.full_model_inference.reset_states()
        if is_save_tf_model:
            import tensorflow as tf
            import keras.backend as K
            frozen_graph = freeze_session(K.get_session(),
                output_names=
                    [out.op.name for out in tracker_model.full_model_inference.outputs])
            tf.train.write_graph(frozen_graph, "tf_graph", "conv_lstm.pb",
                                 as_text=False)

        video_out = "/media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/YOLO_v2_videos/for_Dec21_presentation/multi_frame/" + config['model']['backend'].replace(' ', '') + "_ConvLSTM_" + args.input.split('/')[-1]
        video_reader = cv2.VideoCapture(args.input)
        # ret = video_reader.set(cv2.CAP_PROP_POS_FRAMES, 10300)

        nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

        video_writer = cv2.VideoWriter(video_out,
                               cv2.VideoWriter_fourcc(*'MPEG'), 
                               30.0, 
                               (frame_w, frame_h))

        # for i in tqdm(range(nb_frames)):
        for i in tqdm(list(range(80000))):
            _, image = video_reader.read()
            # if i % 2 == 0: continue
            if i % 3 == 0:
            
                start_time = time()
                # boxes = yolo.predict(image, obj_threshold=obj_thresh)
                boxes = tracker_model.predict_on_image(image,
                                                       obj_threshold=obj_threshold,
                                                       is_filter_bboxes=is_filter_bboxes,
                                                       shovel_type=shovel_type)
                time_taken = time() - start_time
                if False:
                    print(("Time taken: %.3f" % time_taken))
                image = draw_boxes(image, boxes, config['model']['labels'],
                                   score_threshold=obj_threshold)
                video_writer.write(np.uint8(image))

        video_reader.release()
        video_writer.release()  

    else:
        h5_files = os.listdir(args.input)
        # shuffle(h5_files)
        for filename in h5_files:
            filepath = os.path.join(args.input, filename)
            for i in range(config['train']['num_samples_in_h5']):
                tracker_model.predict_on_h5(filepath, i,
                                            args.output,
                                            sequence_length=config['model']['last_sequence_length'],
                                            stride=config['model']['stride'],
                                            obj_threshold=obj_threshold,
                                            nms_threshold=0.01,
                                            is_yolo_pred=True)


argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')
argparser.add_argument('-c', '--conf', type=str, required=True,
        help='path to configuration file')
argparser.add_argument('-i', '--input', type=str, required=True,
        help='path to h5 files or a video')
argparser.add_argument('-o', '--output', type=str, required=True,
        help='path to save images/video.')


if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)
