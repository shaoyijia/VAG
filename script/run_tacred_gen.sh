#!/bin/bash

for ft_task in  $(seq 0 7);
do
  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python finetune_seq2seq.py \
  --max_seq_length 128 \
  --ft_task ${ft_task} \
  --seed $SEED \
  --sequence_file 'sequences/tacred' \
  --baseline $BASELINE \
  --epoch 10 \
  --batch_size 8 --store_ratio $STORE_RATIO --learning_rate 1e-5 --lamb $LAMB --use_dev
done

