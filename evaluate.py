import argparse
import os
import pprint as pp
import json

from frontend import YOLO
from preprocessing import parse_annotation, BatchGenerator


argparser = argparse.ArgumentParser(
    description='Evaluate YOLO_v2 model on any dataset by computing mAP')
argparser.add_argument('-c', '--conf', help='path to configuration file')
argparser.add_argument('-w', '--weights', help='path to pretrained weights')


def _main_(args):
    config_path  = args.conf
    weights_path = args.weights

    with open(config_path) as config_buffer:    
        config = json.load(config_buffer)

    yolo = YOLO(backend             = config['model']['backend'],
                input_size          = config['model']['input_size'], 
                labels              = config['model']['labels'], 
                max_box_per_image   = config['model']['max_box_per_image'],
                anchors             = config['model']['anchors'])

    yolo.load_weights(weights_path)

    valid_imgs, valid_labels = parse_annotation(config['valid']['valid_annot_folder'], 
                                                config['valid']['valid_image_folder'], 
                                                config['model']['labels'])
    yolo.batch_size = config['train']['batch_size']
    yolo.sequence_length = 1
    generator_config = {
        'IMAGE_H'         : yolo.input_size, 
        'IMAGE_W'         : yolo.input_size,
        'GRID_H'          : yolo.grid_h,  
        'GRID_W'          : yolo.grid_w,
        'BOX'             : yolo.nb_box,
        'LABELS'          : yolo.labels,
        'CLASS'           : len(yolo.labels),
        'ANCHORS'         : yolo.anchors,
        'BATCH_SIZE'      : yolo.batch_size,
        'TRUE_BOX_BUFFER' : yolo.max_box_per_image,
        'SEQUENCE_LENGTH' : yolo.sequence_length
    }    
    valid_generator = BatchGenerator(valid_imgs, 
                                 generator_config, 
                                 norm=yolo.feature_extractor.normalize,
                                 jitter=False)   
    ave_precisions = yolo.evaluate(valid_generator, iou_threshold=0.3,
                                   score_threshold=0.2)
    print("ave precisions: ", ave_precisions)
    print('mAP: {:.4f}'.format(sum(ave_precisions.values()) / len(ave_precisions)))         


if __name__ == '__main__':
    args = argparser.parse_args()
    pp.pprint(vars(args))
    _main_(args)
