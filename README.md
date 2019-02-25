# BucketMonitoring Intelligence Series

## What is this repository for?
* This is a linux-python-tensorflow project, for missingToothDetection, and frame selection for FM and WM for the intelligence series.
* this repo is used for developing TM and FM/WM frame selection algorithms. To train the YOLO networks or LSTMs use https://bitbucket.org/motionmetricsdev/yolo-v2
* Version: 1.0.0 (Iteration 1)

## How do I get set up?
1. Create a new Anaconda or venv environment with python 3.5 installed.
    * This can be achieved by running:   conda create -n FMDLAlgoEnv python=3.5

2. Install the requirements specified in the requirements.yml file with pip:
    * This can be achieved by running:    pip3 install -r requirements.yml

3. This project has the following dependencies (aside from either Anaconda or Pip3):
    1. python (3.5): pip3 install python=3.5
    2. tensorflow (1.5 or later): pip3 install tensorflow
    3. keras (any version): pip3 install keras
    4. opencv (either 2 or 3): pip3 install opencv-python
    5. numpy (any version): This is required and will be installed by opencv
    5. pillow: (any version): pip3 install pillow
    6. matplotlib (any version): pip3 install matplotlib



### 1. Data preparation
        1.1 Get the following csv from jira:
            **For Hydraulics:
            project="Data Archive" and "Bucket Bounding Box Images Labeled" = Yes AND "Teeth Images Labeled" = yes AND ("Equipment Type/Model" = Hydraulic ) and "Camera Type" = Optical and "Equipment Type/Model" != "Shovel Model"

            **For Cable:
            project="Data Archive" and "Bucket Bounding Box Images Labeled" = Yes AND "Teeth Images Labeled" = yes AND ("Equipment Type/Model" = "P&H" or "Equipment Type/Model" = Bucyrus ) and "Camera Type" = Optical and "Equipment Type/Model" != "Shovel Model"

            **For Backhoe:
            project="Data Archive" and "Equipment Type/Model" = Backhoe and "Teeth Images Labeled" = Yes and "Scene Images Labeled" = yes and "Camera Type" = Optical

            ### 2. Edit the configuration file
            The configuration file is a json file, which looks like this:


        1.2 The exportJiraImages.py  inside:   Make sure you use cnn_rnn branch for everything.
            https://bitbucket.org/motionmetricsdev/yolo-v2/src/cnn_rnn/data_gen/

            grabs the csv file, saves all the images, and the xmsl for the images. 

            Run the file using:

            python exportJiraImages.py --csv-file '/media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/hydraulic/try11_sameAs5_afterDataCorrections/MMI JIRA 2019-01-14T11_42_49-0800.csv' --output-dir /media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/hydraulic/try11_sameAs5_afterDataCorrections/ --background-dir /media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/hydraulic/try11_sameAs5_afterDataCorrections/backgroundIMages/ --label-type teeth --is-filter-state --is-visible-part --is-filter-visible --is-debug –is-soft-state


        **Output directory must be the same directory wehre the csv file is at. Otherwise you cant find the ImageInfo.xml   if the error persists make the output directory same as where everything (both the script and csv is at). 
        
        Options:
            --is-filter-state     measn exclude state 0. 
            --is-visible-part   Takes only opto the visible part of the tooth for bb not the whole tooth
            -is-filter-visible   This excludes the teeth with not enough visible portion.
            –is-debug       creates a debug folder and saves image with Bbs overlayed. 

        **Above creates one file named ImageInfo.json that has all the bounding boxes for all the images in. 
        And, for each image, it creates an .xml file with the same info as the corresponding entry in json except in VOC format which specifies the formatting of the xml for bounding boxes. Because yolo processes bounding boxes in voc format.  This is the equivalent of the csv files tensorflow uses  with image names, path, xmin,xmax, etc. 


### 2. Edit the configuration file
    The configuration file is a json file, which looks like this:

    ```python
    {
        "model" : {
            "architecture":         "Full Yolo",    # "Tiny Yolo" or "Full Yolo" or "MobileNet" or "SqueezeNet" or "Inception3"
            "input_size":           416,
            "anchors":              [0.57273, 0.677385, 1.87446, 2.06253, 3.33843, 5.47434, 7.88282, 3.52778, 9.77052, 9.16828],
            "max_box_per_image":    10,        
            "labels":               ["raccoon"]
        },

        "train": {
            "train_image_folder":   "/home/andy/data/raccoon_dataset/images/",
            "train_annot_folder":   "/home/andy/data/raccoon_dataset/anns/",      
              
            "train_times":          10,             # the number of time to cycle through the training set, useful for small datasets
            "pretrained_weights":   "",             # specify the path of the pretrained weights, but it's fine to start from scratch
            "batch_size":           16,             # the number of images to read in each batch
            "learning_rate":        1e-4,           # the base learning rate of the default Adam rate scheduler
            "nb_epoch":             50,             # number of epoches
            "warmup_epochs":        3,              # the number of initial epochs during which the sizes of the 5 boxes in each cell is forced to match the sizes of the 5 anchors, this trick seems to improve precision emperically

            "object_scale":         5.0 ,           # determine how much to penalize wrong prediction of confidence of object predictors
            "no_object_scale":      1.0,            # determine how much to penalize wrong prediction of confidence of non-object predictors
            "coord_scale":          1.0,            # determine how much to penalize wrong position and size predictions (x, y, w, h)
            "class_scale":          1.0,            # determine how much to penalize wrong class prediction

            "debug":                true            # turn on/off the line that prints current confidence, position, size, class losses and recall
        },

        "valid": {
            "valid_image_folder":   "",
            "valid_annot_folder":   "",

            "valid_times":          1
        }
    }

    ```

    The model section defines the type of the model to construct as well as other parameters of the model such as the input image size and the list of anchors. The ```labels``` setting lists the labels to be trained on. Only images, which has labels being listed, are fed to the network. The rest images are simply ignored. By this way, a Dog Detector can easily be trained using VOC or COCO dataset by setting ```labels``` to ```['dog']```.


    The config file has info for both yolo and lstm:
    2.1 YOLO backend specification (Full Yolo,  Tiny Yolo, MobileNet, seqeeznet, etc).
    2.2 YOLO  input_size:  This implementation assumes only square images so one number needed here.
    2.4 YOLO  anchors: This are the anchors that gen_anchors.py will created. These specify width and height of each anchor. So in below there are 9 anchor boxes:
    [0.58,0.98,   0.95,1.49,    1.22,2.42,   1.76,2.36,   9.18,2.15,   11.55,2.35,   11.67,5.33,   12.11,8.19,   17.09,12.23]

    2.5 max_box_per_image  this is the maximum number of objects (i.e. bounding boxes) in the image. 

    2.5 YOLO Labels: should match the names in the xml file, and you can exlude labels from here. Below is all the labels we have:
    ["Tooth", "Toothline", "BucketBB", "MatInside", "WearArea", "LipShroud"]

    LSTM related configs:
    2.6 h5_sequence_length
    2.7 last_sequence_length
    2.8 stride
    2.9 detector_weights_path

    2.10 Training images and xml labels are specified in:
                "train_image_folder":   "/media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/data/optical/bucyrus_hydraulic/image/",
            "train_annot_folder":   "/media/sf_N_DRIVE/randd/MachineLearning/TeamMembers/Anuar/data/optical/bucyrus_hydraulic/labels/", 

    2.10   train_h5_folder   is added by Anuar and is not used.
    2.11   train_times   this number just gets multiplied by the number of epochs.
    2.12  pretrained_weights  to start from these weights for finetuning.  If the specified file is not here, it wont use it.

    2.13   batch_size    This specifies the batch size for BOTH yolo and LSTM.
    2.14  num_samples_in_h5   This specifies to LSTM the number of sequences in each h5 input example to lstm
    2.15 learning_rate   this is the initial learning rate for both YOLO and LSTM.  1e-4, is the original. 
    2.16 nb_epochs   total number of epochs for training for both YOLO and LSTM. 
    2.17 warmup_epochs    original impl used 4. 
    2.18 These are the scales of each of the tems in the loss function: Original was 
            "object_scale":         1.0,
            "no_object_scale":      1.0,
            "coord_scale":          1.0,
            "class_scale":          1.0,

    2.19. saved_weights_name This is the name BOTH yolo and lstm use to save their h5 models.
    2.20 debug  should be off when training. Visualizs the images, AFTER augmentaion.
    2.21 is_eval  means whether to calculate MAP on the validation set or not. Turning this on just does MAP without training. 



### 3. Generate anchors for your dataset (optional)

    Now we need to run a script which decides on the best anchors (i.e. coordinates of patches from the images) for us.
    Run the script:  /media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/yolo-v2/gen_anchors.py
    using: 
    python gen_anchors.py -c /media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/yolo-v2/config_full_yolo_new_rnn.json -a 9

    This script takes a path to a config file.  And a number of anchor boxes to generate (-a option). 
    This will give you an array of anchor width and heights that you can copy and paste into the config file. The best anchors are found by running k-means clustering. You can run this multiple times with each label (i.e. tooth vs bucket) to get sets of anchors that are good for each and then add them all. To do this either change the label specification in config file. Or in the gen_anchors.py line 114 provide an array of labels like  [‘teeth’, ‘lipshroud’]  instead of  ['labels']   in below:
        train_imgs, train_labels = parse_annotation(config['train']['train_annot_folder'],
                                                    config['train']['train_image_folder'],
                                                    config['model']['labels'])

### 4. Start the training process

`python train.py -c config.json`

By the end of this process, the code will write the weights of the best model to file best_weights.h5 (or whatever name specified in the setting "saved_weights_name" in the config.json file). The training process stops when the loss on the validation set is not improved in 3 consecutive epoches.

### 5. Perform detection using trained weights on an image by running
    python predict.py -c /media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/hydraulic/try5-NewTrainingProcedure--higherBatchSize/try5-higherBatchSize.json -w /media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/hydraulic/try5-NewTrainingProcedure--higherBatchSize/full_yolo_bb_101_hsBb_valLoss-0.25.h5 -i /media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/hydraulic/testSet-image_hard_pickedByHs/ -o /media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/hydraulic/try5-NewTrainingProcedure—higherBatchSize/preds_onHardTestSet_bestCheckPoint101/

    It carries out detection on the image and write the image with detected bounding boxes to the same folder.



### 6. Creating the Sequence data for training the LSTM heads
    The data for lstm is an h5 file.
    For each image in yolo bb training set we find the previous (unlabeled frames) and pass them trhough the backbone to generate the input feature map for LSTM. We also grab the labels for the last (i.e. labeled) image in the sequence and encode them according to YOLO BB output. 
    run   /media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/yolo-v2/create_seq_data.py

    python create_seq_data.py -i /media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/hydraulic/LSTM/link2CorrectedLabels/ -c /media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/hydraulic/LSTM/hsLSTM_dataGeneration.json -o /media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/hydraulic/LSTM/outputOf_create_seq_data/ -s 16 2> std_err.txt

    This will do all of the above. And give you the h5. 
    **This script needs a config file (same as yolo BB)

    "h5_sequence_length": This is the length of sequence in your h5 (60 means for each labeled frame we take the previous 60 frames with no skips == 2 seconds in the past).
    "last_sequence_length": This is the sequence length we use during training. If you set this to 30, you take the LAST 30 frames of the sequnece in your h5.
    "stride": This is the skip rate for training, if set to 3 it will take every 3 frames in the sequence length specified by last_sequence_length". Stride of 3 == skip 2 frames in the middle. 


    detector_weights_path:   This takes a network and runs inference and saves the results. But right now it saves the final feature map before decoding, where as we want the one to the last layer. 

    train_h5_folder is unused.



    "num_samples_in_h5":    16,   is how many SEQUENCES you want in each h5 file. 


    If you see this error, you need to create a folder in your repo called ‘data’ this is used as a buffer by the script to save pickle files.
    ***Traceback (most recent call last):
      File "create_seq_data.py", line 182, in <module>
        create_seqs(args)
      File "create_seq_data.py", line 121, in create_seqs
        train_batch, valid_batch = load_data_generators(generator_config, args)
      File "create_seq_data.py", line 45, in load_data_generators
        with open(pickle_train, 'wb') as fp:
    FileNotFoundError: [Errno 2] No such file or directory: 'data/TrainAnn_hydr.pickle'




    *******LSTM model is defined in tracker.py,     def create_model_cnn_rnn_extracted(self): function. 


### 7. Training the LSTM heads:
    Run: python tracker.py -c /media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/hydraulic/LSTM/try6__2lstm-256-1b1-30frames/try6__2lstm-256-1b1-30frames.json

### 8. Testing LSTM on H5 test files:
    python tracker_predict.py -c /media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/hydraulic/LSTM/try6__2lstm-256-1b1-30frames/try6__2lstm-256-1b1-30frames.json -i /media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/hydraulic/LSTM/outputOf_create_seq_data_testSet/