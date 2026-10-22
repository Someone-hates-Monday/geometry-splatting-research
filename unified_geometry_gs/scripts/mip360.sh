#!/usr/bin/env bash
dataset_folder=/media/baowen/data/dataset/360_v2
for scene in bicycle garden stump; do
  python train.py -s ${dataset_folder}/${scene} -m output/mip360_sg/${scene} -r 2 --sh_degree 2 --sg_degree 7 --eval
done
