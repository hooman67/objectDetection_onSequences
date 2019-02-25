import cv2
import numpy as np
from frontend import YOLO
from backend import FullYoloFeature
import argparse
import json
import pprint as pp
import os
import sys
import pickle
from time import time
from tqdm import tqdm

# from preprocessing import parse_annotation, BatchSequenceGenerator
from preprocessing_rnn import parse_annotation, BatchSequenceGenerator, SequenceH5Generator

from keras.models import Model
from keras.layers import Reshape, Conv2D, Input, Lambda, ConvLSTM2D
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint, TensorBoard,\
                            ReduceLROnPlateau
import keras.backend as K
from keras.optimizers import Adam, SGD
from keras.layers.wrappers import TimeDistributed
from keras.layers.merge import concatenate
from keras.utils import multi_gpu_model
import tensorflow as tf
import h5py
from utils import decode_netout, draw_boxes
import matplotlib.pyplot as plt
from shutil import copyfile

from callbacks import ValOnlyProgbarLogger, EvaluateCallback, DecayLR,\
        MultiGPUCheckpoint, TrainValTensorBoard_HS

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0,1"


def normalize(self, image):
    return image / 255.


class TrackerModel(object):
    def __init__(self, config, is_inference=False, is_inference_ensemble=False):
        self.image_h = config['model']['input_size']
        self.image_w = config['model']['input_size']
        self.batch_size = config['train']['batch_size']
        self.num_samples_in_h5 = config['train']['num_samples_in_h5']
        self.stride = config['model']['stride']
        self.h5_sequence_length = config['model']['h5_sequence_length']
        self.last_sequence_length = config['model']['last_sequence_length']
        self.sequence_length = self.last_sequence_length / self.stride
        self.true_box_buffer = config['model']['max_box_per_image']

        self.yolo_weights_path = config['model']['detector_weights_path']

        self.nb_box = len(config['model']['anchors']) // 2
        self.anchors = config['model']['anchors']
        self.epochs = config['train']['nb_epochs']
        self.warmup_epochs = config['train']['warmup_epochs']
        self.object_scale = config['train']['object_scale']
        self.no_object_scale = config['train']['no_object_scale']
        self.coord_scale = config['train']['coord_scale']
        self.class_scale = config['train']['class_scale']
        self.labels = config['model']['labels']
        self.nb_class = len(config['model']['labels'])
        self.class_wt = np.ones(self.nb_class, dtype='float32')


        #added by HS HSS
        self.full_log_dir                 = config['train']['full_log_dir']
        self.early_stop_patience          = config['train']['early_stop_patience']
        self.early_stop_min_delta         = config['train']['early_stop_min_delta']
        self.learning_rate_decay_factor   = config['train']['learning_rate_decay_factor']
        self.learning_rate_decay_patience = config['train']['learning_rate_decay_patience']
        self.learning_rate_decay_min_lr   = config['train']['learning_rate_decay_min_lr']
        self.lstm_h5_data_path            = config['train']['lstm_h5_data_path']
        self.saved_weights_name           = config['train']['saved_weights_name']
        ####################



        ############################################
        # Compile the model
        ############################################
        self.detector = YOLO(backend            = config['model']['backend'],
                            input_size          = config['model']['input_size'], 
                            labels              = config['model']['labels'], 
                            max_box_per_image   = config['model']['max_box_per_image'],
                            anchors             = config['model']['anchors'])
        self.detector.load_weights(self.yolo_weights_path)
        self.grid_h, self.grid_w = self.detector.grid_h, self.detector.grid_w

        print('\nhs loaded backbone:')
        pp.pprint(vars(self.detector))



        # self.full_model = self.create_model()
        self.full_model = self.create_model_cnn_rnn_extracted()
        if is_inference:
            if is_inference_ensemble:
                self.full_model_inference = self.create_model_cnn_rnn_extracted_inference_ensemble()
            else:
                self.full_model_inference = self.create_model_cnn_rnn_extracted_inference()

        self.debug = config['train']['debug']
        self.initial_epoch = 0


    def load_weights(self, weights_path):
        self.full_model.load_weights(weights_path)
        print("Successfully loaded weights from %s!" % weights_path)


    def freeze_layers(self, model):
        for layer in model.layers:
            layer.trainable = False
        return model


    def create_model_cnn_rnn_extracted(self):
        input_images = Input(batch_shape=
                (self.batch_size, self.sequence_length, self.image_h, self.image_w, 3),
                name='images_input')
        self.true_boxes = Input(batch_shape=
                (self.batch_size, 1, 1, 1, 1, self.true_box_buffer, 4),
                name='bbox_input')


        #load the detector
        feature_detector_model = self.detector.get_feature_model(
                    is_before_activation=False)
        
        feature_detector_model = self.freeze_layers(feature_detector_model)

        print("\nSummary of feature detector:")
        feature_detector_model.summary()
       


        #run the yolo bb on each image, and stack up the results to be fed to RNN
        yolo_feats_seq = TimeDistributed(feature_detector_model,
                                name='each_frame_feats')\
                                        (input_images)

        recurrent_state = ConvLSTM2D(256, (1, 1), strides=(1, 1), padding='same',
                return_sequences=True, name='conv_lstm_1')(yolo_feats_seq)
        recurrent_state = ConvLSTM2D(256, (1, 1), strides=(1, 1), padding='same',
                return_sequences=False, name='conv_lstm_2')(recurrent_state)
        output_conv = Conv2D(
                self.nb_box * (4 + 1 + self.nb_class), (1,1),
                strides=(1,1), padding='same', kernel_initializer='lecun_normal',
                name='track_conv')(recurrent_state)
        output_reshaped = Reshape(
                (self.grid_h, self.grid_w, self.nb_box,
                    4 + 1 + self.nb_class))(output_conv)

        output_trk = Lambda(lambda args: args[0], name='tracking')(
                        [output_reshaped, self.true_boxes])
        model = Model([input_images, self.true_boxes], output_trk,
                      name='cnn_rnn_model')



        # We initialize the last layer (prediction by lstm) here
        layer = model.layers[-4] # track_conv layer
        weights = layer.get_weights()
        new_kernel = np.random.normal(size=weights[0].shape)/(self.grid_h*self.grid_w)
        new_bias   = np.random.normal(size=weights[1].shape)/(self.grid_h*self.grid_w)
        layer.set_weights([new_kernel, new_bias])

        
        #model.load_weights(self.rnn_weights_path, by_name=True)

        print("\nFull MODEL:")
        model.summary()
        return model#, feature_detector_model


    def set_inference_weights(self):
        for layer in self.full_model.layers:        
            for layer_inference in self.full_model_inference.layers:
                if (layer_inference.name == layer.name):
                    layer_inference.set_weights(layer.get_weights())
                    break
            return


    def create_model_cnn_rnn_extracted_inference(self):
        K.set_learning_phase(0)
        input_images = Input(batch_shape=
                (self.batch_size, 1, self.image_h, self.image_w, 3),
                name='images_input')
        feature_detector_model = self.detector.get_feature_model(
                    is_before_activation=False)

        print("\nSummary of feature detector:")
        feature_detector_model.summary()
        yolo_feats_seq = TimeDistributed(feature_detector_model,
                                name='each_frame_feats')\
                                        (input_images)

        recurrent_state = ConvLSTM2D(256, (1, 1), strides=(1, 1), padding='same',
                return_sequences=True, stateful=True, name='conv_lstm_1')\
                    (yolo_feats_seq)
        recurrent_state = ConvLSTM2D(256, (1, 1), strides=(1, 1), padding='same',
                return_sequences=False, stateful=True, name='conv_lstm_2')\
                    (recurrent_state)
        output_conv = Conv2D(
                self.nb_box * (4 + 1 + self.nb_class), (1,1),
                strides=(1,1), padding='same', kernel_initializer='lecun_normal',
                name='track_conv')(recurrent_state)
        output_reshaped = Reshape(
                (self.grid_h, self.grid_w, self.nb_box,
                    4 + 1 + self.nb_class))(output_conv)
        model = Model(input_images, output_reshaped,
                      name='cnn+rnn model')

        print("\nFull MODEL:")
        model.summary()
        return model


    def load_data_generators(self, generator_config):
        pickle_train = 'data/MultiObjDetTracker_TrainAnn.pickle'
        pickle_val   = 'data/MultiObjDetTracker_ValAnn.pickle'

        if os.path.isfile(pickle_train):
            with open (pickle_train, 'rb') as fp:
               train_imgs = pickle.load(fp)
        else:
            train_imgs, seen_train_labels = parse_annotation(self.train_annot_folder,
                                                             self.train_image_folder,
                                                             labels=self.LABELS)
            with open(pickle_train, 'wb') as fp:
               pickle.dump(train_imgs, fp)

        if os.path.isfile(pickle_val):
            with open (pickle_val, 'rb') as fp:
               valid_imgs = pickle.load(fp)
        else:
            valid_imgs, seen_valid_labels = parse_annotation(self.valid_annot_folder,
                                                             self.valid_image_folder,
                                                             labels=self.LABELS)
            with open(pickle_val, 'wb') as fp:
               pickle.dump(valid_imgs, fp)

        train_batch = BatchSequenceGenerator(train_imgs, generator_config, norm=normalize,
                                             shuffle=True, augment=False)
        valid_batch = BatchSequenceGenerator(valid_imgs, generator_config, norm=normalize,
                                             augment=False)

        return train_batch, valid_batch


    def load_data_generators_seq(self, batch_size):
        #path to folder of H5s is in  self.lstm_h5_data_path
        train_batch = SequenceH5Generator(self.lstm_h5_data_path, batch_size,
                self.num_samples_in_h5, self.last_sequence_length, self.labels,
                stride=self.stride, is_yolo_feats=False, is_augment=True)
        valid_batch = SequenceH5Generator(self.lstm_h5_data_path, batch_size,
                self.num_samples_in_h5, self.last_sequence_length, self.labels,
                stride=self.stride, is_validation=True, is_yolo_feats=False)
        
        return train_batch, valid_batch 


    def custom_loss(self, y_true, y_pred):
        new_shape = self.batch_size 
        y_pred = tf.reshape(y_pred,
                (new_shape, self.grid_h, self.grid_w, self.nb_box, 4 + 1 + self.nb_class))
        y_true = tf.reshape(y_true,
                (new_shape, self.grid_h, self.grid_w, self.nb_box, 4 + 1 + self.nb_class))
        self.true_boxes = tf.reshape(self.true_boxes,
                (new_shape, 1, 1, 1, self.true_box_buffer, 4))

        mask_shape = tf.shape(y_true)[:4]
        
        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(self.grid_w), [self.grid_h]), (1, self.grid_h, self.grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0,2,1,3,4))

        cell_grid = tf.tile(tf.concat([cell_x,cell_y], -1), [self.batch_size, 1, 1, self.nb_box, 1])
        
        coord_mask = tf.zeros(mask_shape)
        conf_mask  = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)
        
        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)
        
        """
        Adjust prediction
        """
        ### adjust x and y      
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid
        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1,1,1,self.nb_box,2])
        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])
        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]
        
        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2] # relative position to the containing cell
        
        ### adjust w and h
        true_box_wh = y_true[..., 2:4] # number of cells accross, horizontally and vertically
        
        ### adjust confidence
        true_wh_half = true_box_wh / 2.
        true_mins    = true_box_xy - true_wh_half
        true_maxes   = true_box_xy + true_wh_half
        
        pred_wh_half = pred_box_wh / 2.
        pred_mins    = pred_box_xy - pred_wh_half
        pred_maxes   = pred_box_xy + pred_wh_half       
        
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)
        
        true_box_conf = iou_scores * y_true[..., 4]
        
        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)
        
        """ Determine the masks """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale
        
        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]
        
        true_wh_half = true_wh / 2.
        true_mins    = true_xy - true_wh_half
        true_maxes   = true_xy + true_wh_half
        
        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)
        
        pred_wh_half = pred_wh / 2.
        pred_mins    = pred_xy - pred_wh_half
        pred_maxes   = pred_xy + pred_wh_half    
        
        intersect_mins  = tf.maximum(pred_mins,  true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]
        
        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores  = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * self.no_object_scale
        
        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * self.object_scale
        
        ### class mask: simply the position of the ground truth boxes (the predictors)
        class_mask = y_true[..., 4] * tf.gather(self.class_wt, true_box_class) * self.class_scale       
        
        """ Warm-up training """
        no_boxes_mask = tf.to_float(coord_mask < self.coord_scale/2.)
        seen = tf.assign_add(seen, 1.)
        
        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, self.warmup_batches+1), 
                              lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask, 
                                       true_box_wh + tf.ones_like(true_box_wh) * \
                                       np.reshape(self.anchors, [1,1,1,self.nb_box,2]) * \
                                       no_boxes_mask, 
                                       tf.ones_like(coord_mask)],
                              lambda: [true_box_xy, 
                                       true_box_wh,
                                       coord_mask])
        
        """ Finalize the loss """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box  = tf.reduce_sum(tf.to_float(conf_mask  > 0.0))
        nb_class_box = tf.reduce_sum(tf.to_float(class_mask > 0.0))
        
        loss_xy    = tf.reduce_sum(tf.square(true_box_xy-pred_box_xy)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh    = tf.reduce_sum(tf.square(true_box_wh-pred_box_wh)     * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf  = tf.reduce_sum(tf.square(true_box_conf-pred_box_conf) * conf_mask)  / (nb_conf_box  + 1e-6) / 2.
        loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)
        loss_class = tf.reduce_sum(loss_class * class_mask) / (nb_class_box + 1e-6)
        
        loss = tf.cond(tf.less(seen, self.warmup_batches+1), 
                      lambda: loss_xy + loss_wh + loss_conf + loss_class + 10,
                      lambda: loss_xy + loss_wh + loss_conf + loss_class)
        
        if self.debug:
            nb_true_box = tf.reduce_sum(y_true[..., 4])
            nb_pred_box = tf.reduce_sum(tf.to_float(true_box_conf > 0.5) * tf.to_float(pred_box_conf > 0.3))
            
            current_recall = nb_pred_box/(nb_true_box + 1e-6)
            total_recall = tf.assign_add(total_recall, current_recall) 

            loss = tf.Print(loss, [loss_xy], message='Loss XY \t', summarize=1000)
            loss = tf.Print(loss, [loss_wh], message='Loss WH \t', summarize=1000)
            loss = tf.Print(loss, [loss_conf], message='Loss Conf \t', summarize=1000)
            loss = tf.Print(loss, [loss_class], message='Loss Class \t', summarize=1000)
            loss = tf.Print(loss, [loss], message='Total Loss \t', summarize=1000)
            loss = tf.Print(loss, [current_recall], message='Current Recall \t', summarize=1000)
            # loss = tf.Print(loss, [total_recall/seen], message='Average Recall \t', summarize=1000)
        
        return loss

    def train(self):
        train_batch, valid_batch = self.load_data_generators_seq(self.batch_size)
        #print('Length of generators:')
        #print(int(len(train_batch)))
        #print(int(len(valid_batch)))
        print("Length of generators: %d, %d" % (len(train_batch), len(valid_batch)))
        [x, b], y = train_batch[0]
        print("Input shapes train: ", x.shape, y.shape, b.shape)
        [x, b], y = valid_batch[0]
        print("Input shapes val: ", x.shape, y.shape, b.shape)
        self.warmup_batches = self.warmup_epochs *\
            (len(train_batch) + len(valid_batch))  / 4 
        print("Using %d warmup batches" % self.warmup_batches)




        ############################################
        # Define your callbacks
        ############################################


        #defined by hs for best val
        checkpoint_multi_hs = MultiGPUCheckpoint(
                        '{name}_{{epoch:02d}}_hsBbAndLstm_valLoss-{{val_loss:.2f}}.h5'.format(name=self.saved_weights_name),
                        verbose=1,
                        save_best_only=True,)


        #defined by HS
        # HS HSS originally i used monitor='val_loss', factor=0.5, patience=20, min_lr=1e-6
        reduce_lr_hs = ReduceLROnPlateau(monitor='val_loss',
                                         factor=self.learning_rate_decay_factor,
                                         patience=self.learning_rate_decay_patience,
                                         min_lr=self.learning_rate_decay_min_lr)




        #HS HSS With a patience of 100 you finish in 200 epochs so I changed it to 400
        early_stop = EarlyStopping(monitor='val_loss', 
                           min_delta=self.early_stop_min_delta, 
                           patience=self.early_stop_patience , 
                           verbose=1)
        


        evaluate_callback_val = EvaluateCallback(valid_batch, self.evaluate)


        decay_lr = DecayLR(32, 40, 0.2)

        optimizer = Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        # optimizer = SGD(lr=1e-4, momentum=0.9, decay=0.0, nesterov=True)






        print("Compiling a model...")
        with tf.device("/cpu:0"):
            self.full_model.compile(loss=self.custom_loss, optimizer=optimizer)
        print("Successfuly compiled the full model!")
        


        ############################################
        # Start the training process
        ############################################  
        steps_per_epoch = len(train_batch)
        print("Using %d/%d for train/val steps_per_epoch of %d batch_size!" %\
                (steps_per_epoch,
                len(valid_batch),
                self.batch_size))



        parallel_model = multi_gpu_model(self.full_model, gpus=2)
        parallel_model.compile(loss=self.custom_loss, optimizer=optimizer)
        
        parallel_model.fit_generator(
                    generator        = train_batch,
                    steps_per_epoch  = steps_per_epoch,
                    epochs           = self.epochs,
                    verbose          = 2 if self.debug else 1,
                    validation_data  = valid_batch,
                    validation_steps = len(valid_batch),

                    callbacks        = [
                        early_stop,
                        checkpoint_multi_hs,
                        TrainValTensorBoard_HS(self.full_log_dir,
                                            write_graph=False, write_images=True),
                        ValOnlyProgbarLogger(verbose=1, count_mode='steps'),
                        reduce_lr_hs],  # reduce_lr
                    

                    workers          = 4,
                    max_queue_size   = 10,
                    use_multiprocessing = True,
                    shuffle          = False,
                    initial_epoch    = self.initial_epoch)


        self.full_model.save(saved_weights_name + "_fullModel_final.h5")


    def evaluate(self, generator, iou_threshold=0.3,score_threshold=0.3,max_detections=100,save_path=None):
            """ Evaluate a given dataset using a given model.
            code originally from https://github.com/fizyr/keras-retinanet

            # Arguments
                generator       : The generator that represents the dataset to evaluate.
                model           : The model to evaluate.
                iou_threshold   : The threshold used to consider when a detection is positive or negative.
                score_threshold : The score confidence threshold to use for detections.
                max_detections  : The maximum number of detections to use per image.
                save_path       : The path to save images with visualized detections to.
            # Returns
                A dict mapping class names to mAP scores.
            """    
            print("\nUsing %.2f IOU and %.2f Score thresholds!" %\
                    (iou_threshold, score_threshold))
            # gather all detections and annotations
            all_detections     = [[None for i in range(generator.num_classes())] for j in range(generator.size())]
            all_annotations    = [[None for i in range(generator.num_classes())] for j in range(generator.size())]

            for i in range(generator.size()):
                if i % 100 == 0: print("%d/%d" % (i, generator.size()))
                raw_image = generator.load_image(i)
                raw_height, raw_width, raw_channels = raw_image.shape

                pred_boxes, filtered_boxes  = self.predict(raw_image,
                        obj_threshold=score_threshold, is_filter_bboxes=False)

                score = np.array([box.score for box in pred_boxes])
                pred_labels = np.array([box.label for box in pred_boxes])        
                
                if len(pred_boxes) > 0:
                    pred_boxes = np.array([[box.xmin*raw_width, box.ymin*raw_height,
                        box.xmax*raw_width, box.ymax*raw_height, box.score]
                            for box in pred_boxes])
                else:
                    pred_boxes = np.array([[]])  
                
                # sort the boxes and the labels according to scores
                score_sort = np.argsort(-score)
                pred_labels = pred_labels[score_sort]
                pred_boxes  = pred_boxes[score_sort]
                
                # copy detections to all_detections
                for label in range(generator.num_classes()):
                    all_detections[i][label] = pred_boxes[pred_labels == label, :]
                    
                annotations = generator.load_annotation(i)
                
                # copy detections to all_annotations
                for label in range(generator.num_classes()):
                    all_annotations[i][label] = annotations[annotations[:, 4] == label, :4].copy()
                    
            # compute mAP by comparing all detections and all annotations
            average_precisions = {}
            
            for label in range(generator.num_classes()):
                false_positives = np.zeros((0,))
                true_positives  = np.zeros((0,))
                scores          = np.zeros((0,))
                num_annotations = 0.0

                for i in range(generator.size()):
                    detections           = all_detections[i][label]
                    annotations          = all_annotations[i][label]
                    num_annotations     += annotations.shape[0]
                    detected_annotations = []

                    for d in detections:
                        scores = np.append(scores, d[4])

                        if annotations.shape[0] == 0:
                            false_positives = np.append(false_positives, 1)
                            true_positives  = np.append(true_positives, 0)
                            continue

                        overlaps            = compute_overlap(np.expand_dims(d, axis=0), annotations)
                        assigned_annotation = np.argmax(overlaps, axis=1)
                        max_overlap         = overlaps[0, assigned_annotation]

                        if max_overlap >= iou_threshold and assigned_annotation not in detected_annotations:
                            false_positives = np.append(false_positives, 0)
                            true_positives  = np.append(true_positives, 1)
                            detected_annotations.append(assigned_annotation)
                        else:
                            false_positives = np.append(false_positives, 1)
                            true_positives  = np.append(true_positives, 0)

                # no annotations -> AP for this class is 0 (is this correct?)
                if num_annotations == 0:
                    average_precisions[label] = 0
                    continue

                # sort by score
                indices         = np.argsort(-scores)
                false_positives = false_positives[indices]
                true_positives  = true_positives[indices]

                # compute false positives and true positives
                false_positives = np.cumsum(false_positives)
                true_positives  = np.cumsum(true_positives)

                # compute recall and precision
                recall    = true_positives / num_annotations
                precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)

                # compute average precision
                average_precision  = compute_ap(recall, precision)  
                average_precisions[label] = average_precision

            return average_precisions    

    def predict(self, image):
        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.input_size, self.input_size))
        image = np.divide(image, 255., dtype=np.float32)

        input_image = image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)
        dummy_array = np.zeros((1,1,1,1,self.max_box_per_image,4))

        netout = self.full_model.predict([input_image, dummy_array])[0]
        boxes  = decode_netout(netout, self.detector.anchors, self.detector.nb_class)

        return boxes

    def predict_on_image(self, image, obj_threshold=0.3, nms_threshold=0.01,
        is_inference_ensemble=False, is_filter_bboxes=False,
                         shovel_type="Hydraulic"):
        image_h, image_w, _ = image.shape
        image = cv2.resize(image, (self.image_h, self.image_w))
        image = np.divide(image, 255., dtype=np.float32)

        input_image = image[:,:,::-1]
        input_image = np.expand_dims(input_image, 0)
        input_image = np.expand_dims(input_image, 0)
        netout = self.full_model_inference.predict(input_image)
        boxes = decode_netout(netout[0, ...], self.anchors, self.nb_class,
                              obj_threshold=obj_threshold,
                              nms_threshold=nms_threshold)
        if is_filter_bboxes:
            boxes = filter_teeth_teethline(boxes, obj_threshold, shovel_type)

        return boxes

    def predict_on_h5(self, h5_path, idx, path_to_save, sequence_length=30, stride=1,
                      obj_threshold=0.3, nms_threshold=0.1, is_yolo_pred=False):

        f = h5py.File(h5_path, 'r')
        x_batches = f["x_batches"]
        if is_yolo_pred:
            yolo_outs = f["yolo_out"]
        b_batches = f["b_batches"]
        y_batches = f["y_batches"]

        id_in_h5 = idx % x_batches.shape[0]

        x_batch = x_batches[id_in_h5, ...] # read from disk
        start_time = time()
        x_batch = np.divide(x_batch, 255., dtype=np.float32)
        print("time: %.3f" % (time() - start_time))
        x_batch = x_batch[..., ::-1]
        if is_yolo_pred:
            yolo_out = yolo_outs[id_in_h5, ...]
        b_batch = b_batches[id_in_h5, ...]
        y_batch = y_batches[id_in_h5, ...]

        #                [reverse  ][:seq_length:skip ][reverse]
        x_batch = x_batch[::-1, ...][:sequence_length:stride][::-1, ...]
        if is_yolo_pred:
            yolo_out = yolo_out[::-1, ...][:sequence_length:stride][::-1, ...]
        image = x_batch[-1, ...].copy()
        image_yolo = image.copy()
        image_first = x_batch[0, ...].copy()
        b_batch = b_batch[::-1, ...][:sequence_length:stride][::-1, ...]
        x_batch = np.expand_dims(x_batch, axis=0)
        if is_yolo_pred:
            yolo_out = np.expand_dims(yolo_out, axis=0)
        b_batch = np.expand_dims(b_batch, axis=0)
        y_batch = np.expand_dims(y_batch, axis=0)

        # x_batch = yolo_out
        start_time = time()
        netouts = self.full_model.predict([x_batch, b_batch])
        print("Time taken ConvLSTM: %.3f" % (time() - start_time))
        print("image.shape: ", image.shape)
        # boxes_yolo = self.detector.predict(image, obj_threshold=obj_threshold-0.2)
        if is_yolo_pred:
            netouts_yolo_last = yolo_out[:, -1, ...]
            netouts_yolo_first = yolo_out[:, 0, ...]
        labels_tensor = y_batch[0, ...].copy()

        for i,netout in enumerate(netouts):
            start_time = time()
            boxes  = decode_netout(netout, self.anchors, self.nb_class,
                                   obj_threshold, nms_threshold)
            print("Decoding time: %.3f" % (time() - start_time))
            if is_yolo_pred:
                boxes_yolo_first = decode_netout(netouts_yolo_first[i], self.anchors,
                                           self.nb_class, obj_threshold, nms_threshold) 
                boxes_yolo_last = decode_netout(netouts_yolo_last[i], self.anchors,
                                           self.nb_class, obj_threshold, nms_threshold) 
            labels_boxes = decode_netout(labels_tensor, self.anchors,
                                       self.nb_class, obj_threshold, nms_threshold) 

            image_labels = image.copy()
            image_conv_lstm = draw_boxes(image, boxes, self.labels, obj_threshold)
            if is_yolo_pred:
                image_yolo_first = draw_boxes(image_first, boxes_yolo_first,
                                            self.labels, obj_threshold)
                image_yolo_last = draw_boxes(image_yolo, boxes_yolo_last,
                                            self.labels, obj_threshold)
            image_labels = draw_boxes(image_labels, labels_boxes,
                                        self.labels, obj_threshold)
            print("Bounding boxes found %d/%d" %\
                    (len(boxes), len(labels_boxes)))

        #fig, ax = plt.subplots(1, 3, figsize=(19, 10))
        #fig.tight_layout()
        #if is_yolo_pred:
         #   ax[0].imshow(image_yolo_first)
          #  ax[1].imshow(image_yolo_last)
        #ax[2].imshow(image_conv_lstm)
        #ax[0].set_title("First Frame YOLO pred")
        #ax[1].set_title("Last Frame YOLO pred")
        #ax[2].set_title("Conv LSTM")
        # plt.show()
        #plt.imshow(image_conv_lstm)
        #plt.savefig("/media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/hydraulic/LSTM/try6__2lstm-256-1b1-30frames/preds_onHardTestSet/pred_" + h5_name + "_" +  str(idx) + ".jpg",
                    #format='jpg')


        h5_name = h5_path.split('/')[-1]
        filepath = os.path.join(path_to_save, "pred_" + h5_name + str(idx) + ".jpg")
        temps = image_conv_lstm*255. #np.uint8(image_conv_lstm*255)
        cv2.imwrite(filepath, temps) 
        # plt.figure(figsize=(10, 10))
        # plt.imshow(image_conv_lstm)
        # plt.savefig("./lstm_preds/pred_" + h5_name + "_" +  str(idx) + ".jpg",
        #             format='jpg')
        # plt.show()

        f.close()
        
        return



def main(args):
    config_path = args.conf
    #copies the config to where the models at 
    #dst_path = os.path.join("/media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/yolo_v2_models/RNN/", config_path.split('/')[-1])
    #copyfile(config_path, dst_path)
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())
    pp.pprint(config)




    ###########HS setting GPU flags##########################
    #from keras.backend.tensorflow_backend import set_session
    #sessconfig = tf.ConfigProto()
    #sessconfig.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
    #sessconfig.gpu_options.per_process_gpu_memory_fraction=0.1
    #sessconfig.log_device_placement = True  # to log device placement (on which device the operation ran)
                                        # (nothing gets printed in Jupyter, only if you run it standalone)
    #sess = tf.Session(config=sessconfig)
    #set_session(sess)

    ######################################################################


    tracker_model = TrackerModel(config)
    pp.pprint(vars(tracker_model))
    tracker_model.train()

argparser = argparse.ArgumentParser(
    description='Train and validate YOLO_v2 model on any dataset')

argparser.add_argument(
    '-c',
    '--conf',
    help='path to configuration file')


if __name__ == '__main__':
    args = argparser.parse_args()
    main(args)

