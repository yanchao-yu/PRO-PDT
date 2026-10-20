#!/bin/bash

for dataset in vggface2; do
  for model in Resnet Vit; do
      if [[$model == "Inception_resnet"]]
      then
        sbatch train_resnet.sh $dataset

      elif [[$model == "Vit"]]
      then
        sbatch train_vit.sh $dataset
      fi
  done
done