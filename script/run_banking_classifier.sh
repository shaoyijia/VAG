#!/bin/bash

for ft_task in  $(seq 0 6);
do
  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python finetune_classifier.py \
  --max_seq_length 128 \
  --ft_task ${ft_task} \
  --seed $SEED \
  --sequence_file 'sequences/banking77' \
  --baseline $BASELINE \
  --epoch $EPOCH \
  --batch_size 8 --learning_rate 1e-5 --classifier_lr 1e-5 \
  --lamb $LAMB --store_ratio $STORE_RATIO --use_dev
done
