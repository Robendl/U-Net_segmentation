# Semi-Supervised training of U-Net for medical image segmentation

This repository contains all the files that were by Group DLP20 used for assignment 2 from the Deep Learning course. The brain tumour dataset that was used can be found [here](https://www.kaggle.com/datasets/nikhilroxtomar/brain-tumor-segmentation).

Below the purpose of each file is described.

`crop_and_resize.py`: Contains code used for data preprocessing. \
`convert_data_brain.py`: Performs data preprocessing and saves the labelled/unlabelled train/test splits.\
`augmentations.py`: Used during training for data augmentations.\
`network.py`: Used to initialize the U-Net architecture.\
`simclr.py`: Contains self-supervised learning procedure SimCLR.\
`unet.py`: Contains supervised learning procedure of the U-Net using k-fold validaiton.\
`test.py`: Used to get evaluation scores.

`gridsearch.py`: Code used for hyperparameter tuning of the supervised learning procedure.