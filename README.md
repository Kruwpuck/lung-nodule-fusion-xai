# lung-nodule-fusion-xai

## Config

Active config: `configs/config.yaml` — every `src/stage_*.py` script defaults `--config` to this file. It defines the canonical 6-backbone set (`mobilenetv3_small, efficientnet_b0, densenet121, resnet50, vgg16, vit_base`) actually used for the 120 trained checkpoints (6 backbones × 4 arms × 5 folds).

`configs/train.yaml` was an earlier draft with a different, unused backbone set; it is archived at `docs/archive/train.yaml.deprecated` and not read by any script.