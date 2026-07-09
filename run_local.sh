#!/bin/bash
set -e
CFG=configs/config.yaml

python src/stage_00_preprocess.py --config $CFG
python src/stage_01_radiomics.py  --config $CFG
python src/stage_02_split.py      --config $CFG

for model in mobilenetv3_small efficientnet_b0 densenet121 resnet50 vgg16 vit_base; do
  for fold in 0 1 2 3 4; do
    python src/stage_03_train.py --config $CFG --model $model --fold $fold
  done
done

python src/stage_04_evaluate.py --config $CFG
python src/stage_05_xai.py      --config $CFG
