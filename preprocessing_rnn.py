import os
import copy
import xml.etree.ElementTree as ET
import sys
import time
import gc
from random import shuffle

from video_processor import VideoProcessor
from frame_match_corrector import FrameMatchCorrector
from utils import BoundBox, bbox_iou

import pprint as pp
import matplotlib.pyplot as plt
import cv2
import h5py
import numpy as np
import imgaug as ia
from imgaug import augmenters as iaa
from keras.utils import Sequence


def parse_annotation(ann_dir, img_dir, labels=[]):
    print("\nStarting annotation parsing...")
    all_imgs = []
    seen_labels = {}
    
    for ann in sorted(os.listdir(ann_dir)):
        img = {'object':[]}

        try:
            tree = ET.parse(ann_dir + ann)
        except Exception as e:
            print("Skipping image %s due to:\n%s" % (ann, e))
            continue
        
        for elem in tree.iter():
            if 'filename' in elem.tag:
                img['filename'] = img_dir + elem.text
            if 'video_path' in elem.tag:
                img['video_path'] = elem.text
            if 'width' in elem.tag:
                img['width'] = int(elem.text)
            if 'height' in elem.tag:
                img['height'] = int(elem.text)
            if 'object' in elem.tag or 'part' in elem.tag:
                obj = {}
                
                for attr in list(elem):
                    if 'name' in attr.tag:
                        obj['name'] = attr.text

                        if obj['name'] in seen_labels:
                            seen_labels[obj['name']] += 1
                        else:
                            seen_labels[obj['name']] = 1
                        
                        if len(labels) > 0 and obj['name'] not in labels:
                            break
                        else:
                            img['object'] += [obj]
                            
                    if 'bndbox' in attr.tag:
                        for dim in list(attr):
                            if 'xmin' in dim.tag:
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))

        if len(img['object']) > 0:
            all_imgs += [img]
    print("seen_labels: \n")
    pp.pprint(seen_labels)
    print("Finished reading annotations!")
    return all_imgs, seen_labels


class BatchGenerator(Sequence):
    def __init__(self, images, 
                       config, 
                       shuffle=True, 
                       jitter=True, 
                       norm=None):
        self.generator = None

        self.images = images
        self.config = config

        self.shuffle = shuffle
        self.jitter  = jitter
        self.norm    = norm

        self.anchors = [BoundBox(0, 0, config['ANCHORS'][2*i], config['ANCHORS'][2*i+1])\
                for i in range(int(len(config['ANCHORS'])//2))]

        ### augmentors by https://github.com/aleju/imgaug
        sometimes = lambda aug: iaa.Sometimes(0.33, aug)

        # Define our sequence of augmentation steps that will be applied to every image
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                iaa.Fliplr(0.5), # horizontally flip 50% of all images
                #iaa.Flipud(0.2), # vertically flip 20% of all images
                sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                sometimes(iaa.Affine(
                    scale={"x": (0.85, 1.15), "y": (0.85, 1.15)}, # scale images to 80-120% of their size, individually per axis
                    translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, # translate by -20 to +20 percent (per axis)
                    rotate=(-5, 5), # rotate by -45 to +45 degrees
                    #shear=(-5, 5), # shear by -16 to +16 degrees
                    #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                    [
                        #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 3.0
                            # iaa.AverageBlur(k=(2, 5)), # blur image using local means with kernel sizes between 2 and 7
                            # iaa.MedianBlur(k=(3, 6)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        # search either for all edges or for directed edges
                        #sometimes(iaa.OneOf([
                        #    iaa.EdgeDetect(alpha=(0, 0.7)),
                        #    iaa.DirectedEdgeDetect(alpha=(0, 0.7), direction=(0.0, 1.0)),
                        #])),
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.04*255), per_channel=0.5), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 5% of the pixels
                            #iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                        ]),
                        iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.Multiply((0.75, 1.25), per_channel=0.5), # change brightness of images (50-150% of original value)
                        iaa.ContrastNormalization((0.7, 1.3), per_channel=0.5), # improve or worsen the contrast
                        #iaa.Grayscale(alpha=(0.0, 1.0)),
                        #sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
                        #sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))) # sometimes move parts of the image around
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )

        if shuffle: np.random.shuffle(self.images)

    def __len__(self):
        return int(np.ceil(float(len(self.images))//self.config['BATCH_SIZE']))   

    def num_classes(self):
        return len(self.config['LABELS'])

    def size(self):
        return len(self.images)    

    def load_annotation(self, i):
        annots = []

        for obj in self.images[i]['object']:
            annot = [obj['xmin'], obj['ymin'], obj['xmax'], obj['ymax'],
                     self.config['LABELS'].index(obj['name'])]
            annots += [annot]

        if len(annots) == 0: annots = [[]]

        return np.array(annots)

    def load_image(self, i):
        return cv2.imread(self.images[i]['filename'])


    def __getitem__(self, idx):
        l_bound = idx*self.config['BATCH_SIZE']
        r_bound = (idx+1)*self.config['BATCH_SIZE']

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - self.config['BATCH_SIZE']

        instance_count = 0

        x_batch = np.zeros((r_bound - l_bound, self.config['IMAGE_H'],
                            self.config['IMAGE_W'], 3))  # input images
        b_batch = np.zeros((r_bound - l_bound, 1     , 1     , 1    ,
                            self.config['TRUE_BOX_BUFFER'], 4))  # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'], 
                            self.config['GRID_W'], self.config['BOX'],
                            4+1+len(self.config['LABELS'])))  # desired network output

        for train_instance in self.images[l_bound:r_bound]:
            pp.pprint(train_instance)
            img, all_objs = self.aug_image(train_instance, jitter=self.jitter)
            
            # construct output from object's x, y, w, h
            true_box_index = 0
            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and\
                        obj['name'] in self.config['LABELS']:
                    center_x = .5*(obj['xmin'] + obj['xmax'])
                    center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                    center_y = .5*(obj['ymin'] + obj['ymax'])
                    center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

                    grid_x = int(np.floor(center_x))
                    grid_y = int(np.floor(center_y))

                    if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                        obj_indx  = self.config['LABELS'].index(obj['name'])
                        
                        center_w = (obj['xmax'] - obj['xmin']) /\
                                (float(self.config['IMAGE_W']) / self.config['GRID_W']) # unit: grid cell
                        center_h = (obj['ymax'] - obj['ymin']) /\
                                (float(self.config['IMAGE_H']) / self.config['GRID_H']) # unit: grid cell
                        
                        box = [center_x, center_y, center_w, center_h]
                        # find the anchor that best predicts this box
                        best_anchor = -1
                        max_iou     = -1
                        shifted_box = BoundBox(0, 
                                               0,
                                               center_w,  
                                               center_h)
                        for i in range(len(self.anchors)):
                            anchor = self.anchors[i]
                            iou    = bbox_iou(shifted_box, anchor)
                            if max_iou < iou:
                                best_anchor = i
                                max_iou     = iou
                                
                        # assign ground truth x, y, w, h, confidence and class probs to y_batch
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 0:4] = box
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 4  ] = 1.
                        y_batch[instance_count, grid_y, grid_x, best_anchor, 5+obj_indx] = 1
                        
                        # assign the true box to b_batch
                        b_batch[instance_count, 0, 0, 0, true_box_index] = box
                        true_box_index += 1
                        true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']
                            
            # assign input image to x_batch
            if self.norm != None: 
                x_batch[instance_count] = self.norm(img)
            else:
                # plot image and bounding boxes for sanity check
                for obj in all_objs:
                    if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                        cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']),
                                      (obj['xmax'],obj['ymax']), (255,0,0), 3)
                        cv2.putText(img[:,:,::-1], obj['name'], 
                                    (obj['xmin']+2, obj['ymin']+12), 
                                    0, 1.2e-3 * img.shape[0], 
                                    (0,255,0), 2)
                        
                x_batch[instance_count] = img
            # increase instance counter in current batch
            instance_count += 1  

        return [x_batch, b_batch], y_batch


    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)


    def aug_image(self, train_instance, jitter):
        image_name = train_instance['filename']
        image = cv2.imread(image_name)

        if image is None: 
            print('Cannot find ', image_name)
            print("Trying again...")
            image = cv2.imread(image_name)
            if image is None:
                print("Still cannot fine image")
                raise ValueError()

        h, w, c = image.shape
        all_objs = copy.deepcopy(train_instance['object'])

        if jitter:
            ### scale the image
            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(image, (0,0), fx = scale, fy = scale)

            ### translate the image
            max_offx = (scale-1.) * w
            max_offy = (scale-1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)
            image = image[offy : (offy + h), offx : (offx + w)]

            ### flip the image
            flip = np.random.binomial(1, .5)
            if flip > 0.5: image = cv2.flip(image, 1)
            image = self.aug_pipe.augment_image(image)            
            
        # resize the image to standard size
        image = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W']))
        image = image[:,:,::-1]

        # fix object's position and size
        for obj in all_objs:
            for attr in ['xmin', 'xmax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offx)
                obj[attr] = int(obj[attr] * float(self.config['IMAGE_W']) / w)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_W']), 0)
                
            for attr in ['ymin', 'ymax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offy)
                obj[attr] = int(obj[attr] * float(self.config['IMAGE_H']) / h)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_H']), 0)

            if jitter and flip > 0.5:
                xmin = obj['xmin']
                obj['xmin'] = self.config['IMAGE_W'] - obj['xmax']
                obj['xmax'] = self.config['IMAGE_W'] - xmin
                
        return image, all_objs

def normalize(image):
    return image / 255.


class BatchSequenceGenerator(BatchGenerator):
    def __init__(self, images, config, shuffle=True, augment=True, norm=None):
        self.images = images
        self.config = config
        self.shuffle = shuffle
        # self.norm = normalize
        self.norm = norm
        self.jitter = augment
        self.anchors = [BoundBox(0, 0, config['ANCHORS'][2*i], config['ANCHORS'][2*i+1])\
                for i in range(int(len(config['ANCHORS'])//2))]
        self.idx = 0


    def on_epoch_end(self):
        if self.shuffle: np.random.shuffle(self.images)


    def aug_image(self, train_instance, jitter):
        image_name = train_instance['filename']
        image = cv2.imread(image_name)
        if image is None: 
            print('Cannot find ', image_name)
            print("Trying again...")
            image = cv2.imread(image_name)
            if image is None:
                print("Still cannot find image")
                raise ValueError()

        h, w, c = image.shape
        all_objs = copy.deepcopy(train_instance['object'])

        if jitter:
            ### scale the image
            scale = np.random.uniform() / 10. + 1.
            image = cv2.resize(image, (0,0), fx = scale, fy = scale)

            ### translate the image
            max_offx = (scale-1.) * w
            max_offy = (scale-1.) * h
            offx = int(np.random.uniform() * max_offx)
            offy = int(np.random.uniform() * max_offy)
            image = image[offy : (offy + h), offx : (offx + w)]

            ### flip the image
            flip = np.random.binomial(1, .5)
            if flip > 0.5: image = cv2.flip(image, 1)
            image = self.aug_pipe.augment_image(image)            
            
        # resize the image to standard size
        image = cv2.resize(image, (self.config['IMAGE_H'], self.config['IMAGE_W']))
        image = image[:,:,::-1]

        # fix object's position and size
        for obj in all_objs:
            for attr in ['xmin', 'xmax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offx)
                obj[attr] = int(obj[attr] * float(self.config['IMAGE_W']) / w)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_W']), 0)
                
            for attr in ['ymin', 'ymax']:
                if jitter: obj[attr] = int(obj[attr] * scale - offy)
                obj[attr] = int(obj[attr] * float(self.config['IMAGE_H']) / h)
                obj[attr] = max(min(obj[attr], self.config['IMAGE_H']), 0)

            if jitter and flip > 0.5:
                xmin = obj['xmin']
                obj['xmin'] = self.config['IMAGE_W'] - obj['xmax']
                obj['xmax'] = self.config['IMAGE_W'] - xmin
                
        return image, all_objs


    def get_input_sequence(self, train_instance, idx):
        sequence_length = self.config['SEQUENCE_LENGTH']
        image_path = train_instance['filename']
        video_path = train_instance['video_path']
        num_frames_before_after = [-self.config['SEQUENCE_LENGTH'],
                                    self.config['SEQUENCE_LENGTH']]
        video_proc = VideoProcessor(num_frames_before_after, enhance_clahe=False)
        frame_number = int(image_path.split("_")[-1].split(".")[0])
        cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
        frames, frame_names = video_proc.get_video_frames(cap, frame_number)
        if frames is None:
            print("No frames!")
            return None

        frame_corrector = FrameMatchCorrector(frames, image_path)
        frames = frame_corrector.correct_frame_match()
        if frames is None:
            # print("Non-matched!")
            return None
        # else:
            # print "Matched!"
        frames = frames[:, : int(frames.shape[1]/2) + 1, ...]
        frames = frames[0, -self.config['SEQUENCE_LENGTH']:]

        if self.norm is not None:
            for i in range(frames.shape[0]):
                frames[i] = self.norm(frames[i])    

        return frames


    def get_output(self, train_instance, idx):
        b_batch = np.zeros((1     , 1     , 1    ,  self.config['TRUE_BOX_BUFFER'], 4))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((self.config['GRID_H'],  self.config['GRID_W'],
                            self.config['BOX'], 4+1+len(self.config['LABELS'])))
        img, all_objs = self.aug_image(train_instance, jitter=False)
        # construct output from object's x, y, w, h
        true_box_index = 0
        for obj in all_objs:
            if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin'] and\
                    obj['name'] in self.config['LABELS']:
                center_x = .5*(obj['xmin'] + obj['xmax'])
                center_x = center_x / (float(self.config['IMAGE_W']) / self.config['GRID_W'])
                center_y = .5*(obj['ymin'] + obj['ymax'])
                center_y = center_y / (float(self.config['IMAGE_H']) / self.config['GRID_H'])

                grid_x = int(np.floor(center_x))
                grid_y = int(np.floor(center_y))

                if grid_x < self.config['GRID_W'] and grid_y < self.config['GRID_H']:
                    obj_indx  = self.config['LABELS'].index(obj['name'])
                    
                    center_w = (obj['xmax'] - obj['xmin']) / (float(self.config['IMAGE_W']) / self.config['GRID_W']) # unit: grid cell
                    center_h = (obj['ymax'] - obj['ymin']) / (float(self.config['IMAGE_H']) / self.config['GRID_H']) # unit: grid cell
                    
                    box = [center_x, center_y, center_w, center_h]
                    # find the anchor that best predicts this box
                    best_anchor = -1
                    max_iou     = -1
                    shifted_box = BoundBox(0, 
                                           0,
                                           center_w,  
                                           center_h)
                    
                    for i in range(len(self.anchors)):
                        anchor = self.anchors[i]
                        iou    = bbox_iou(shifted_box, anchor)
                        if max_iou < iou:
                            best_anchor = i
                            max_iou     = iou
                            
                    # assign ground truth x, y, w, h, confidence and class probs to y_batch
                    y_batch[grid_y, grid_x, best_anchor, 0:4] = box
                    y_batch[grid_y, grid_x, best_anchor, 4  ] = 1.
                    y_batch[grid_y, grid_x, best_anchor, 5+obj_indx] = 1
                    
                    # assign the true box to b_batch
                    b_batch[0, 0, 0, true_box_index] = box
                    
                    true_box_index += 1
                    true_box_index = true_box_index % self.config['TRUE_BOX_BUFFER']

        is_debug = False 
        if is_debug:
            for obj in all_objs:
                if obj['xmax'] > obj['xmin'] and obj['ymax'] > obj['ymin']:
                    cv2.rectangle(img[:,:,::-1], (obj['xmin'],obj['ymin']),
                                  (obj['xmax'],obj['ymax']), (255,0,0), 1)
                    cv2.putText(img[:,:,::-1], obj['name'], 
                                (obj['xmin']+2, obj['ymin']+12), 
                                0, 1.2e-3 * img.shape[0], 
                                (0,255,0), 1)
                    filepath = "/home/cheng/Desktop/repos/object-tracking/utility/debug_data/" + train_instance['filename'].split('/')[-1]
                    ret = cv2.imwrite(filepath, img)
            
        return b_batch, y_batch


    def post_process_frames(self, frames):
        new_frames = np.zeros(((frames.shape[0],) + (self.config['IMAGE_H'], self.config['IMAGE_W'], 3)))
        for i, frame in enumerate(frames):
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
            frame = cv2.resize(frame, (self.config['IMAGE_H'], self.config['IMAGE_W']), interpolation=cv2.INTER_NEAREST)

            new_frames[i, ...] = frame 
        return new_frames

    def __getitem__(self, idx):
        batch_size = self.config['BATCH_SIZE']
        sequence_length = self.config['SEQUENCE_LENGTH']
        l_bound = self.idx * batch_size
        r_bound = (self.idx+1) * batch_size
        # print "idx: %d, batch_size: %d, sequence_length: %d" % (idx, batch_size, sequence_length)

        if r_bound > len(self.images):
            r_bound = len(self.images)
            l_bound = r_bound - batch_size

        instance_count = 0
        x_batch = np.zeros((r_bound - l_bound, sequence_length-5,
            self.config['IMAGE_H'], self.config['IMAGE_W'], 3))  
        b_batch = np.zeros((r_bound - l_bound, 1, 1, 1, 1, 
            self.config['TRUE_BOX_BUFFER'], 4))   # list of self.config['TRUE_self.config['BOX']_BUFFER'] GT boxes
        y_batch = np.zeros((r_bound - l_bound, self.config['GRID_H'],
                            self.config['GRID_W'], self.config['BOX'],
                            4+1+len(self.config['LABELS'])))    # desired network output

        # for i, train_seq in enumerate(self.images[l_bound:r_bound]):
        i, j = 0, 0
        while j != r_bound - l_bound:
            train_seq = self.images[l_bound + i]
            self.seed = np.random.randint(low=0, high=10000) + idx

            x_instance = self.get_input_sequence(train_seq, self.idx)
            i += 1
            self.idx += 1
            if x_instance is None:
                # print("Skipping a batch, ",)
                continue
            x_instance = self.post_process_frames(x_instance)
            x_batch[j,:,:,:,:] = x_instance

            b_instance, y_instance = self.get_output(train_seq, self.idx)
            b_batch[j,:,:,:,:,:] = b_instance
            y_batch[j,:,:,:,:]   = y_instance
            instance_count += 1
            j += 1

        is_all_zeros = not np.any(x_batch) or not np.any(y_batch)
        if is_all_zeros:
            print(("Batch wrong: zeros!", is_all_zeros))
        return [x_batch, b_batch], y_batch


class SequenceH5Generator(Sequence):
    def __init__(self, input_path, batch_size, num_samples_in_h5,
            sequence_length, labels, stride=1, is_debug=False,
            is_validation=False, is_yolo_feats=True, is_augment=False):
        self.input_path = input_path
        self.batch_size = batch_size
        self.num_in_h5 = num_samples_in_h5
        self.sequence_length = sequence_length
        self.labels = labels
        self.stride = stride
        self.is_debug = is_debug
        self.is_validation = is_validation
        self.is_yolo_feats = is_yolo_feats
        assert(self.num_in_h5 % self.batch_size == 0)
        self.num_batches_in_h5 = self.num_in_h5 / self.batch_size

        filepaths = os.listdir(self.input_path)
        shuffle(filepaths)
        if not is_validation: 
            self.filepaths = filepaths[:-3]
        else:
            self.filepaths = filepaths[-3:]

        self.is_augment = is_augment
        sometimes = lambda aug: iaa.Sometimes(0.35, aug)
        # All augmenters with per_channel=0.5 will sample one value _per image_
        # in 50% of all cases. In all other cases they will sample new values
        # _per channel_.
        self.aug_pipe = iaa.Sequential(
            [
                # apply the following augmenters to most images
                # iaa.Fliplr(0.5), # horizontally flip 50% of all images
                #iaa.Flipud(0.2), # vertically flip 20% of all images
                # sometimes(iaa.Crop(percent=(0, 0.1))), # crop images by 0-10% of their height/width
                # sometimes(iaa.Affine(
                    # scale={"x": (0.85, 1.15), "y": (0.85, 1.15)}, # scale images to 80-120% of their size, individually per axis
                    # translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, # translate by -20 to +20 percent (per axis)
                    # rotate=(-5, 5), # rotate by -45 to +45 degrees
                    #shear=(-5, 5), # shear by -16 to +16 degrees
                    #order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
                    #cval=(0, 255), # if mode is constant, use a cval between 0 and 255
                    #mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
                # )),
                # execute 0 to 5 of the following (less important) augmenters per image
                # don't execute all of them, as that would often be way too strong
                iaa.SomeOf((0, 5),
                    [
                        #sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(20, 200))), # convert images into their superpixel representation
                        iaa.OneOf([
                            iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 3.0
                            # iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                            # iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                        ]),
                        # iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                        #iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.04*255), per_channel=0.5), # add gaussian noise to images
                        iaa.OneOf([
                            iaa.Dropout((0.01, 0.05), per_channel=0.5), # randomly remove up to 10% of the pixels
                        ]),
                        iaa.Invert(0.05, per_channel=True), # invert color channels
                        iaa.Add((-7, 7), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                        iaa.Multiply((0.8, 1.25), per_channel=0.5), # change brightness of images (50-150% of original value)
                        iaa.ContrastNormalization((0.8, 1.25), per_channel=0.5), # improve or worsen the contrast
                        #iaa.Grayscale(alpha=(0.0, 1.0)),
                    ],
                    random_order=True
                )
            ],
            random_order=True
        )


    def __len__(self):
        return int(self.num_in_h5 * len(self.filepaths) // self.batch_size)

    def num_classes(self):
        return len(self.labels)

    def size(self):
        return self.num_in_h5 * len(self.filepaths)


    def aug_batch(self, x_batch):
        rand_batch_seed = np.random.randint(1000)
        for i_seq in range(x_batch.shape[0]):
            ia.seed(rand_batch_seed + i_seq)
            for j_img in range(x_batch.shape[1]):
                single_image = x_batch[i_seq, j_img, ...]
                x_batch[i_seq, j_img, ...] = self.aug_pipe.augment_image(single_image)
        return x_batch


    def __getitem__(self, idx):
        gc.collect()
        filename_idx = int(idx / self.num_batches_in_h5)
        filename = self.filepaths[filename_idx]
        filepath = os.path.join(self.input_path, filename)

        try:
            # print "\nCurrent file: ", filename
            f = h5py.File(filepath, 'r')
        except Exception as e:
            print(("Error: ", e, ". File: ", filepath))
            filename = self.filepaths[filename_idx + 1]
            filepath = os.path.join(self.input_path, filename)
            f = h5py.File(filepath, 'r')

        x_batches = f["x_batches"]
        b_batches = f["b_batches"]
        y_batches = f["y_batches"]
        l_bound_in_h5 = int(idx % self.num_batches_in_h5 * self.batch_size)
        r_bound_in_h5 = int(idx % self.num_batches_in_h5 * self.batch_size + self.batch_size)

        x_batch = x_batches[int(l_bound_in_h5) : int(r_bound_in_h5), ...] # read from disk
        if x_batch.shape[0] == 0:
            print("\nError 0-dim axis! ", x_batch.shape)
            print("idx = %d; l_bound_in_h5 = %d; r_bound_in_h5=%d; filename = %s;\
                filename_idx=%d" %\
                    (idx, l_bound_in_h5, r_bound_in_h5, filename, filename_idx))
            l_bound_in_h5 = 0
            r_bound_in_h5 = self.batch_size
            x_batch = x_batches[l_bound_in_h5 : r_bound_in_h5, ...] # read from disk

        if self.is_augment:
            x_batch = self.aug_batch(x_batch) # TODO: Check before norm or after????
        x_batch = np.divide(x_batch, 255., dtype=np.float32)
        b_batch = b_batches[int(l_bound_in_h5) : int(r_bound_in_h5), ...]
        y_batch = y_batches[int(l_bound_in_h5) : int(r_bound_in_h5), ...]
        x_batch = x_batch[:, ::-1, ...][:, :self.sequence_length:self.stride][:, ::-1, ...]
        b_batch = b_batch[:, ::-1, ...][:, :self.sequence_length:self.stride][:, ::-1, ...]

        if self.sequence_length == 1:
            x_batch = np.squeeze(x_batch, axis=0)
            b_batch = np.squeeze(b_batch, axis=0)

        f.close()
        gc.collect()

        return [x_batch, b_batch], y_batch


        
