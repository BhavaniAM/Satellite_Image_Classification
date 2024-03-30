# Satellite Image Classification

This repository explains the method of Image Classification on satellite images to classify/group them using Convolutional Neural Networks into 4 different classes: Barren land, Forest, Grassland and Other (a class that consists of all land cover classes other than the previous three).

The dataset used is DeepSat4, which is taken from the [DeepSat](https://www.kaggle.com/datasets/crawford/deepsat-sat4) notebook on Kaggle. 

## Description from the Kaggle repository

Originally, the images were extracted from the National Agriculture Imagery Program (NAIP) dataset. The NAIP dataset consists of a total of 330,000 scenes spanning the whole of the Continental United States (CONUS). The average image tiles are ~6000 pixels in width and ~7000 pixels in height, measuring around 200 megabytes each. The entire NAIP dataset for CONUS is ~65 terabytes. 

The images consist of 4 bands - red, green, blue and Near Infrared (NIR). In order to maintain the high variance inherent in the entire NAIP dataset, we sample image patches from a multitude of scenes (a total of 1500 image tiles) covering different landscapes like rural areas, urban areas, densely forested, mountainous terrain, small to large water bodies, agricultural areas, etc. covering the whole state of California. An image labeling tool developed as part of this study was used to manually label uniform image patches belonging to a particular landcover class.

Once labeled, 28x28 non-overlapping sliding window blocks were extracted from the uniform image patch and saved to the dataset with the corresponding label. We chose 28x28 as the window size to maintain a significantly bigger context, and at the same time not to make it as big as to drop the relative statistical properties of the target class conditional distributions within the contextual window. Care was taken to avoid interclass overlaps within a selected and labeled image patch.

## Model, Training and Testing


400,000 images were used to train the model and 100,000 images were used as the test set. The corresponding ground truth labels are also provided. Sample images for each class are shown below:


A convolutional neural network was used to train the model. To run the code and reproduce the results, clone the repository on your system or on Colab and run:
```
python main.py
```
![Barren Land](/images/barren_land.png)
![Forest Land](/images/forest_land.png)
![Grassland](/images/grassland.png)
![Other](/images/other.png)
The ipynb file can be directly run on Colab using the 'Open in Colab' option.
