#!/usr/bin/env bash
dataset_folder=/media/baowen/data/dataset/tnt_official
for scene in Barn Caterpillar Ignatius Truck Meetingroom Courthouse; do
  python train.py -s ${dataset_folder}/${scene} -m output/tnt/${scene} -r 2 --use_decoupled_appearance 3 --data_device cuda
  python mesh_extract_tetrahedra.py -m output/tnt/${scene}
  python eval_tnt/run.py --dataset-dir ${dataset_folder}/${scene} --traj-path ${dataset_folder}/${scene}/${scene}_COLMAP_SfM.log --ply-path output/tnt/${scene}/recon_post.ply --out-dir output/tnt/${scene}/mesh
done
