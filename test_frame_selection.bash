#!/bin/bash

declare -a arr=(
	"/home/hooman/videos/1084_Ahafo_Hydraulic_liebherr_R9400/2018012214/1_20180122-151700_0001n0.avi")


# now loop through the above array
for i in "${arr[@]}"
do
    echo "$i"
	python predict.py -i $i -w /media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/hydraulic/try11_sameAs5_afterDataCorrections/full_yolo_bb_final.h5 -c /media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/hydraulic/try11_sameAs5_afterDataCorrections/try11_sameAs5_afterDataCorrections.json -o /media/hooman/1tb-ssd-hs3-linu/BucketTracking-Project/hydraulic/frameSelection/try1/ --frame_select
done
