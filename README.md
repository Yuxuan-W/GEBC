# GEB+: A Benchmark for Generic Event Boundary Captioning, Grounding and Retrieval

![image](https://github.com/Yuxuan-W/GEB-Plus/blob/master/figures/Cover.png)
We introduce a new dataset called **Kinetic-GEB+**. The dataset consists of over 170k boundaries associated with captions describing status changes in the generic events in 12K videos. Upon this new dataset, we propose three tasks supporting the development of a more fine-grained, robust, and human-like understanding of videos through status changes. 
We evaluate many representative baselines in our dataset, where we also design a new **TPD (Temporal-based Pairwise Difference) Modeling** method for visual difference and achieve significant performance improvements. Besides, the results show there are still formidable challenges for current methods in the utilization of different granularities, representation of visual difference, and the accurate localization of status changes. Further analysis shows that our dataset can drive developing more powerful methods to understand status changes and thus improve video level comprehension.

## Preparing evaluation package
Download the package https://github.com/LuoweiZhou/coco-caption/tree/de6f385503ac9a4305a1dcdc39c02312f9fa13fc/pycocoevalcap and put it under `utils` folder as:

`GEBC/utils/pycocoevalcap`

## Preparing features
Download and unzip the features from our listed google drive link: https://drive.google.com/drive/folders/1E-KML1rU_gd6CF4nkkNG8Jm3Cq6VBYRR?usp=sharing, make sure you have the following path:

`GEBC/datasets/features/region_feature`

`GEBC/datasets/features/tsn_captioning_feature`

## Run the baseline
To run the captioning baseline, execute the following command:

`python run_captioning.py --do_train --ablation obj --evaluate_during_training`

You can customize the argument following the annotation, but note that we do not provide annotation of testset so it might cause trouble to use  `--do_test` config.
