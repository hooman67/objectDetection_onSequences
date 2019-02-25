import os, sys
import pickle
import numpy as np
import h5py
import argparse
import json
import pprint as pp

from preprocessing import parse_annotation
from preprocessing_rnn import BatchSequenceGenerator, normalize
from frontend import YOLO

argparser = argparse.ArgumentParser(
    description='Create H5 files with image and label sequences')
argparser.add_argument('-i', '--input',
        help="Path to input folders including a folder 'image' and 'labels'")
argparser.add_argument('-c', '--conf', help='path to configuration file')
argparser.add_argument('-o', '--output_path', help="path to output directory")
argparser.add_argument('-w', '--weights', help='path to pretrained weights')
argparser.add_argument('-s', '--size_of_h5', type=int, default=16,
        help='number of sequence samples in 1 h5 file')
argparser.add_argument('-y', '--is_yolo_feats', action='store_true', 
        help='whether to output yolo features or not')
argparser.add_argument('-t', '--is_test_data', action='store_true', 
        help='whether to generate test data (instead of train data)')

"""
python create_seq_data.py -i ... -c config_full_yolo_new_rnn.json -o /media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/sequence_data/YOLO_RNN/full_h5s_new_10_anchors  --size_of_h5=16 --is_yolo_feats --is_test_data
"""

def load_data_generators(generator_config, args):
    pickle_train = 'data/TrainAnn_hydr.pickle'
    pickle_val   = 'data/ValAnn_hydr.pickle'
    train_image_folder = os.path.join(args.input, "image/")
    train_annot_folder = os.path.join(args.input, "labels/")
    valid_image_folder = train_image_folder
    valid_annot_folder = train_annot_folder
    if os.path.isfile(pickle_train):
        print("\n=== WARNING!!! Opening an old pickled file!!! ===")
        with open (pickle_train, 'rb') as fp:
           train_imgs = pickle.load(fp)
    else:
        train_imgs, seen_train_labels = parse_annotation(train_annot_folder,
                train_image_folder, labels=generator_config['LABELS'])
        with open(pickle_train, 'wb') as fp:
           pickle.dump(train_imgs, fp)

    if os.path.isfile(pickle_val):
        with open (pickle_val, 'rb') as fp:
           valid_imgs = pickle.load(fp)
    else:
        valid_imgs, seen_valid_labels = parse_annotation(valid_annot_folder,
                                valid_image_folder, labels=generator_config['LABELS'])
        with open(pickle_val, 'wb') as fp:
           pickle.dump(valid_imgs, fp)

    train_batch = BatchSequenceGenerator(train_imgs, generator_config, norm=None,
                                         shuffle=True, augment=False)
    valid_batch = BatchSequenceGenerator(valid_imgs, generator_config, norm=None,
                                         augment=False)
    return train_batch, valid_batch

def load_detector():

    return detector

def get_yolo_features(yolo, x_batch, generator_config):
    sequence_length = x_batch.shape[0]
    netouts = np.zeros((sequence_length, generator_config['GRID_H'],
                        generator_config['GRID_W'], generator_config['BOX'],
                        4 + 1 + generator_config['CLASS']),
                        dtype=np.float32)
    for i_seq in range(sequence_length):
        image = x_batch[i_seq, ...]
        image = image / 255.
        image = np.expand_dims(image, axis=0)

        dummy_array = np.zeros((1,1,1,1,generator_config['TRUE_BOX_BUFFER'],4))
        netout = yolo.model.predict([image, dummy_array])
        netouts[i_seq, ...] = netout

    return netouts


def create_seqs(args):
    output_path = args.output_path
    config_path = args.conf
    with open(config_path) as config_buffer:    
        config = json.loads(config_buffer.read())
    pp.pprint(config)

    IMAGE_H = config['model']['input_size']
    IMAGE_W = config['model']['input_size']
    GRID_H, GRID_W = int((IMAGE_H / 32)), int((IMAGE_W / 32))
    assert(IMAGE_H / 32. == GRID_H)
    NB_BOX = int((len(config['model']['anchors']) / 2))
    LABELS = config['model']['labels']
    ANCHORS = config['model']['anchors']
    BATCH_SIZE = config['train']['batch_size']
    if BATCH_SIZE != 1:
        print("Changing batch size to 1!")
        BATCH_SIZE = 1
    SEQUENCE_LENGTH = config['model']['h5_sequence_length']
    CLASS = len(LABELS)
    MAX_BOX_PER_IMAGE = config['model']['max_box_per_image']

    generator_config = {
        'IMAGE_H'         : IMAGE_H,
        'IMAGE_W'         : IMAGE_W,
        'GRID_H'          : GRID_H,
        'GRID_W'          : GRID_W,
        'BOX'             : NB_BOX,
        'LABELS'          : LABELS,
        'CLASS'           : len(LABELS),
        'ANCHORS'         : ANCHORS,
        'BATCH_SIZE'      : BATCH_SIZE,
        'TRUE_BOX_BUFFER' : MAX_BOX_PER_IMAGE,
        'SEQUENCE_LENGTH' : SEQUENCE_LENGTH+5
    }

    train_batch, valid_batch = load_data_generators(generator_config, args)
    print("Length of Generators", len(train_batch), len(valid_batch))
    if args.is_test_data:
        print("\nCreating test data!")
        train_batch = valid_batch

    if args.is_yolo_feats:
        yolo = YOLO(backend             = config['model']['backend'],
                    input_size          = config['model']['input_size'], 
                    labels              = config['model']['labels'], 
                    max_box_per_image   = config['model']['max_box_per_image'],
                    anchors             = config['model']['anchors'])
        yolo.load_weights(args.weights)


    num_of_h5s = int((len(train_batch) / args.size_of_h5))
    print("\nCreating %d h5 files...\n" % num_of_h5s)
    for i in range(num_of_h5s):
        x_batches = np.zeros((0, SEQUENCE_LENGTH, IMAGE_H, IMAGE_W, 3), dtype=np.uint8)
        if args.is_yolo_feats:
            yolo_out  = np.zeros((0, SEQUENCE_LENGTH, GRID_H, GRID_W, NB_BOX, 4+1+CLASS),
                                 dtype=np.float32)
        b_batches = np.zeros((0, 1, 1, 1, 1, MAX_BOX_PER_IMAGE, 4), dtype=np.float32)
        y_batches = np.zeros((0, GRID_H, GRID_W, NB_BOX, 4 + 1 + CLASS), dtype=np.float32)
        print("Doing %d-th h5 file." % i)
        for j in range(args.size_of_h5):
            [x_batch, b_batch], y_batch = train_batch[i*args.size_of_h5 + j]
            if not x_batch.any() or not b_batch.any() or not y_batch.any():
                print("\nWRWROOOOONG!\n")
                continue

            # these are the images
            x_batches = np.concatenate((x_batches, x_batch), axis=0)
            b_batches = np.concatenate((b_batches, b_batch), axis=0)
            
            # these are the GT labels encoded into YOLO outputMap format
            y_batches = np.concatenate((y_batches, y_batch), axis=0)
            del y_batch

            if args.is_yolo_feats:
                # features for the whole sequence_length
                netouts = get_yolo_features(yolo, x_batch[0, ...], generator_config)
                netouts = np.expand_dims(netouts, axis=0)
                yolo_out = np.concatenate((yolo_out, netouts), axis=0)

        filename = "sequences_%d.h5" % i
        h5path = os.path.join(output_path, filename)
        # comp_kwargs = {'compression': 'gzip', 'compression_opts': 1} # default is 4
        with h5py.File(h5path, 'w') as f:
            f.create_dataset('x_batches', data=x_batches, dtype='uint8')#, **comp_kwargs)
            if args.is_yolo_feats:
                f.create_dataset('yolo_out', data=yolo_out, dtype='float32')#, **comp_kwargs)
            f.create_dataset('b_batches', data=b_batches, dtype='float32')#, **comp_kwargs)
            f.create_dataset('y_batches', data=y_batches, dtype='float32')#, **comp_kwargs)
    return


if __name__ == '__main__':
    args = argparser.parse_args()
    output_path = args.output_path
    pp.pprint(vars(args))
    create_seqs(args)
