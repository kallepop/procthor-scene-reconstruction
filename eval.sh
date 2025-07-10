#!/bin/bash
CASE_NAME="proc2dat"
house_id="2"

# path to ground truth masks and bounding boxes
gt_folder="/home/ckaese/Documents/LangSplat/datasets"

root_path="../"
python evaluate_iou_loc.py \
        --dataset_name ${CASE_NAME} \
        --feat_dir ${root_path}/output \
        --ae_ckpt_dir ${root_path}/autoencoder/ckpt \
        --output_dir ${root_path}/eval_result \
        --mask_thresh 0.4 \
        --encoder_dims 256 128 64 32 3 \
        --decoder_dims 16 32 64 128 256 256 512 \
        --grtr_folder ${gt_folder} \
        --h_id ${house_id}
