#! /usr/bin/env python

import os
import sys

import argparse
from tqdm import tqdm
import json
import pprint as pp

import cv2
import numpy as np

import keras
from keras import backend as K
import tensorflow as tf


from frame_selection import FmFrameSelector, RandomFmFrameSelector
from frontend import YOLO
from xml_utils import gen_xml_file
from utils import *


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

obj_thresh = 0.2
class_obj_threshold = [0.5, 0.3, 0.3, 0.3, 0.3, 0.3]
shovel_type = "Hydraulic"
if shovel_type == "Cable": num_teeth = 8
elif shovel_type == "Hydraulic": num_teeth = 6
elif shovel_type == "Backhoe": num_teeth = 6
else: raise ValueError


argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')
argparser.add_argument('-c', '--conf', required=True, help='path to configuration file')
argparser.add_argument('-w', '--weights', required=True, help='path to pretrained weights')
argparser.add_argument('-i', '--input', required=True,
        help='path to an image or a video (mp4 format)')
argparser.add_argument('-o', '--output', type=str, required=True,
        help='path to save images/video.')
argparser.add_argument('-s', '--save_soft', action='store_true',
        help='whether to save soft predicted labels in xml file (only for images)')
argparser.add_argument('-f', '--frame_select', action='store_true',
        help='whether to select frames for WM and FM')




def getFmFrameSelectors(video_path, output_path, frame_rate, skip_rate, stride):
    #Make a folder to save results at
    save_folder_selections = os.path.join(output_path,
                                          video_path.split('/')[-1][:-4])
    if not os.path.exists(save_folder_selections):
        os.mkdir(save_folder_selections)


    # Buffering Parameters in seconds
    maxLength_frameBuffer_seconds = 70 # max length of frame_buffer and bbox_buffer
    frequeny_decisionMaking_seconds = 45 # every K-th seconds a decision is made. (i.e. we run our algo and attempt to select frames if we can)
    timeWindow_selectFrame_seconds = 45 # time window that's taken in select_frame()
    print("maxLength_frameBuffer_seconds = %d, frequeny_decisionMaking_seconds = %d, timeWindow_selectFrame_seconds = %d" % (maxLength_frameBuffer_seconds, frequeny_decisionMaking_seconds, timeWindow_selectFrame_seconds))
    
    # effective parameters taking into account skip_rate and frame_rate
    maxLength_frameBuffer, frequeny_decisionMaking = maxLength_frameBuffer_seconds * frame_rate // skip_rate, frequeny_decisionMaking_seconds * frame_rate 
    timeWindow_selectFrame = timeWindow_selectFrame_seconds * frame_rate // skip_rate
    print("maxLength_frameBuffer = %d, frequeny_decisionMaking = %d, timeWindow_selectFrame = %d" % (maxLength_frameBuffer, frequeny_decisionMaking, timeWindow_selectFrame))


    # None = missing detection. 
    # if we have less than these many detections no selection is made.
    minNumber_validScoresToSlectFrame = 15


    fm_frame_selector = FmFrameSelector(
        maxLength_frameBuffer,
        frequeny_decisionMaking,
        timeWindow_selectFrame,
        save_folder_selections,
        label_ind=3,
        stride=stride,
        minNumber_validScoresToSlectFrame=minNumber_validScoresToSlectFrame)


    fm_random_frame_selector = RandomFmFrameSelector(
        maxLength_frameBuffer,
        frequeny_decisionMaking,
        timeWindow_selectFrame,
        save_folder_selections,
        label_ind=3,
        stride=stride,
        minNumber_validScoresToSlectFrame=minNumber_validScoresToSlectFrame,
        selection_option="random")

    fm_weighted_frame_selector = RandomFmFrameSelector(
        maxLength_frameBuffer,
        frequeny_decisionMaking,
        timeWindow_selectFrame,
        save_folder_selections,
        label_ind=3,
        stride=stride,
        minNumber_validScoresToSlectFrame=minNumber_validScoresToSlectFrame,
        selection_option="weighted")

    return fm_frame_selector, fm_random_frame_selector, fm_weighted_frame_selector


def predictOnVideo(video_path, output_path, config, maxFramesToRunOn, runFrameSelection, detectionModel):    
    
    video_out_name = config['model']['backend'] + "_" +\
                     video_path.split('/')[-1][:-4] + ".avi"
    
    video_out = os.path.join(output_path, video_out_name)
    
    video_reader = cv2.VideoCapture(video_path)
    

    nb_frames = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_h = int(video_reader.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_w = int(video_reader.get(cv2.CAP_PROP_FRAME_WIDTH))

    #hs won't process any frames after this
    frames_to_predict = min(maxFramesToRunOn, nb_frames-30)

    skip_rate = 3
    frame_rate = 30


    video_writer = cv2.VideoWriter(video_out,
                           cv2.VideoWriter_fourcc(*'MPEG'), 
                           frame_rate / skip_rate, 
                           (frame_w, frame_h))



    if runFrameSelection:
        log = open("frame_selection_results.txt", 'w')
        fm_frame_selector, fm_random_frame_selector, fm_weighted_frame_selector = getFmFrameSelectors(video_path, output_path, frame_rate, skip_rate, config['model']['stride'])



    bboxes_in_all_frames = []
    labels = config['model']['labels']

    for i in tqdm(list(range(frames_to_predict))):
        _, image = video_reader.read()

        if i % skip_rate == 0:

            boxes, filtered_boxes = detectionModel.predict(image, obj_threshold=obj_thresh,
                                 nms_threshold=0.01, is_filter_bboxes=False,
                                 shovel_type=shovel_type,
                                 class_obj_threshold=class_obj_threshold)
          
            #we do this to draw both original and filtered boxes. Draw boxes knows to draw them with different thicknesses.
            boxes += filtered_boxes
                     
            image = draw_boxes(image, boxes,
                               config['model']['labels'],
                               score_threshold=obj_thresh,
                               class_obj_threshold=class_obj_threshold)



            if runFrameSelection:
                # wm_frame_selector.update(i, image, boxes, num_teeth,
                #                          write_selection=True)
                fm_frame_selector.update(i, image, boxes, num_teeth,
                                         write_selection=True)
                # wm_random_frame_selector.update(i, image, boxes,
                #                                 write_selection=True)
                fm_random_frame_selector.update(i, image, boxes,
                                                write_selection=True)
                # wm_weighted_frame_selector.update(i, image, boxes,
                #                                 write_selection=True)
                fm_weighted_frame_selector.update(i, image, boxes,
                                                write_selection=True)



            video_writer.write(np.uint8(image))
            bboxes_in_all_frames.append(boxes)



    video_reader.release()
    video_writer.release()  




    if runFrameSelection:

        frame_selection_rate = float(len(fm_frame_selector.frames_selected)) /\
                (float(frames_to_predict) / (frame_rate*60.)) 
       
        print("\n", video_path, file=log)
        print("%d frames were selected in %d-frame video with %.1f imgs/min rate"\
                % (len(fm_frame_selector.frames_selected), frames_to_predict,
                frame_selection_rate), file=log)


        #Path to where the time stamp labels are. 
        #evaluate_frame_selection("/home/hooman/randd/MachineLearning/TeamMembers/Farhad/timestamp_labels/Scene/", fm_frame_selector.selected_frame_numbers, log)


def predictOnImageDir(image_dir_path, output_path, config,savePredictionsAsXmlToo, detectionModel):
    image_dir_paths = []
    labels = config['model']['labels']

    #get image paths
    if os.path.isdir(image_dir_path): 
        for inp_file in os.listdir(image_dir_path):
            image_dir_paths += [os.path.join(image_dir_path, inp_file)]
    else:
        image_dir_paths += [image_dir_path]

    image_dir_paths = [inp_file for inp_file in image_dir_paths if\
            (inp_file[-4:] in ['.jpg', '.png', 'JPEG'])]



    if savePredictionsAsXmlToo:
        #setup the directory to save at
        parent_dir = os.path.abspath(os.path.join(output_path, os.pardir))
        path_to_write_xml = os.path.join(parent_dir, "soft_bucket_labels")
        if not os.path.exists(path_to_write_xml):
            os.mkdir(path_to_write_xml)



    for image_dir_path in image_dir_paths:
        print(("Processing ", image_dir_path))

        
        #load and check the image, if not good continue with the other images
        image = cv2.imread(image_dir_path)
        if image is None:
            image = cv2.imread(image_dir_path)
            if image is None:
                print("Couldn't read image jumping to next!")
                continue

        

        boxes, filtered_boxes = detectionModel.predict(image, obj_threshold=obj_thresh,
                             nms_threshold=0.01, is_filter_bboxes=False,
                             shovel_type=shovel_type)
        
        boxes += filtered_boxes


        #visualize the predictions
        image = draw_boxes(image, boxes, labels,
                           score_threshold=obj_thresh) 


        #save the results
        path_to_save = os.path.join(output_path, image_dir_path.split('/')[-1])
        cv2.imwrite(path_to_save, np.uint8(image))         

        if savePredictionsAsXmlToo:
            gen_xml_file(image_dir_path, boxes, labels, path_to_write_xml,
                         excluded_classes=["Tooth", "Toothline"])


def predictOnH5Dir(h5s_dir_path, output_path, config, obj_threshold, detectionModel):
    h5_files = os.listdir(h5s_dir_path)

    for file_name in h5_files:
        filepath = os.path.join(h5s_dir_path, file_name)
        print("Predicting file %s" % first_file_name)
        
        for i in range(config['train']['num_samples_in_h5']):
            yolo.predict_on_h5(filepath, i,
                           output_path,
                           sequence_length=config['model']['last_sequence_length'],
                           stride=config['model']['stride'],
                           obj_threshold=obj_threshold,
                           nms_threshold=0.01)


def _main_(args):
    with open(args.conf) as config_buffer:    
        config = json.load(config_buffer)



    ###############################
    #   Load the model 
    ###############################
    # keras.backend.set_session(K.tf.Session(config=K.tf.ConfigProto(intra_op_parallelism_threads=8, inter_op_parallelism_threads=8)))

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    yolo.load_weights(args.weights)



    ###############################
    #   Decide what to predict on
    ###############################
    if args.input[-4:] in ['.mp4', '.avi', '.mov']:
        predictOnVideo(args.input, args.output, config, maxFramesToRunOn=5000, runFrameSelection=args.frame_select, detectionModel=yolo)

    else:
        first_file_name = os.listdir(args.input)[0]

        # predict on folder of images and save images with overlayed bboxes
        if first_file_name[-4:] in [".jpg", ".png"]:
            predictOnImageDir(args.input, args.output, config, savePredictionsAsXmlToo=args.save_soft, detectionModel=yolo)


        elif first_file_name[-4:] in [".h5", ".hdf5"]:
            predictOnH5Dir(args.input, args.output, config, obj_threshold=obj_thresh, detectionModel=yolo)

        else:
            print('input -i argument extension did not match what we expected')




if __name__ == '__main__':
    args = argparser.parse_args()
    pp.pprint(vars(args))
    _main_(args)
