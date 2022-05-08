# GEBC

## Preparing evaluation package
Download the package https://github.com/LuoweiZhou/coco-caption/tree/de6f385503ac9a4305a1dcdc39c02312f9fa13fc/pycocoevalcap and put it under `utils` folder as:

`GEBC/utils/pycocoevalcap`

## Preparing features
Download and unzip the features from our listed google drive link, make sure you have the following path:

`GEBC/datasets/features/region_feature`

`GEBC/datasets/features/tsn_captioning_feature`

## Run the baseline
To run the captioning baseline, execute the following command:

`python run_captioning.py --do_train --ablation obj --evaluate_during_training`

You can customize the argument following the annotation, but note that we do not provide annotation of testset so it might cause trouble to use  `--do_test` config.
