# GEB+: A Benchmark for Generic Event Boundary Captioning, Grounding and Retrieval
[GEB+: A Benchmark for Generic Event Boundary Captioning, Grounding and Retrieval](https://arxiv.org/abs/2204.00486) [[PDF](https://arxiv.org/pdf/2204.00486.pdf)]

[Yuxuan Wang](https://yuxuanw.me), [Difei Gao](https://scholar.google.com/citations?user=No9OsocAAAAJ&hl=en), [Licheng Yu](https://lichengunc.github.io), Stan Weixian Lei, Matt Feiszli, and [Mike Zheng Shou](https://sites.google.com/view/showlab)

We introduce a new dataset called **Kinetic-GEB+**. The dataset consists of over 170k boundaries associated with captions describing status changes in the generic events in 12K videos. Upon this new dataset, we propose three tasks (**Boundary Captioning**, **Boundary Grounding** and **Boundary Caption-Video Retrieval**) supporting the development of a more fine-grained, robust, and human-like understanding of videos through status changes.

![image](https://github.com/Yuxuan-W/GEB-Plus/blob/master/figures/Cover.png)

We evaluate many representative baselines in our dataset, where we also design a new **TPD (Temporal-based Pairwise Difference) Modeling** method for visual difference and achieve significant performance improvements. Besides, the results show there are still formidable challenges for current methods in the utilization of different granularities, representation of visual difference, and the accurate localization of status changes. Further analysis shows that our dataset can drive developing more powerful methods to understand status changes and thus improve video level comprehension.

![image](https://github.com/Yuxuan-W/GEB-Plus/blob/master/figures/Tasks.png)

## Using Kinetic-GEBC Dataset
In our **Kinetic-GEB+** dataset, each video contains 1 to 8 annotations from different annotators and each annotation consists of several boundaries inside a video, where the boundaries' location are not the same. 
In the evaluation of downstream tasks, we select one annotator whose labeled boundaries are most consistent with others to reduce noise and duplication. Then, we use these boundaries’ timestamps as the anchors to merge other annotators’ captions, preserving the diversity of different opinions. Thus, one video corresponds to multiple boundaries, and each boundary could be with multiple captions. Finally, this selection includes 40k anchors from all videos.

**Here we release 2 versions of dataset:**

a) **[Filtered datasets](https://drive.google.com/drive/folders/1KlFQO__GuUlue_4uCj5oBzzx357D9WE_?usp=sharing) (Recommended)** used in our paper, which has been adjusted for downstream task, including 40K filtered boundaries.

b) **[Raw annotation](https://drive.google.com/drive/folders/1ZEoqsr9gy4FluRhSQgIBp2NiYWVIhcsQ?usp=sharing)** that could be used as supplement in training your own model, including 170K boundaries.

Note that our paper uses version a) in the evaluation of our model, please also evaluated your own model with version a) in future comparisons.

## Using Our Baseline Models

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
