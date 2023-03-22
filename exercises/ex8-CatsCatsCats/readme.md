# Exercise 8 - Cats, Cats, Cats (WORK IN PROGRESS)

Are you sad that you have watched all cat movies and seen all cat photos on the internet? Then be sad no more - in this exercise we will make a *Cat Synthesizer* where you can create all the cat photos you will ever need!

Also, you can help your friends with missing cats to find the perfect new *twin cat*.
 
To be able to do these wonderful things we will harness the power of image based *principal component analysis*. The methods we will use, can be called *classical machine learning*.


## Learning Objectives

After completing this exercise, the student should be able to do the following:

1. Use the python function `glob` to find all files with a given pattern in a folder.

## Importing required Python packages

We will use the virtual environment from the previous exercise (`course02502`). 

Let us start with some imports:

```python
from skimage import io
from skimage.util import img_as_ubyte
from skimage.util import img_as_float
import matplotlib.pyplot as plt
import numpy as np
import glob
from sklearn.decomposition import PCA
from skimage.transform import SimilarityTransform
from skimage.transform import warp
import os
```

## Exercise data and material

The data and material needed for this exercise can be found here: 
[exercise data and material](https://github.com/RasmusRPaulsen/DTUImageAnalysis/tree/main/exercises/ex8-CatsCatsCats/data)

The main part of the data is a large database of photos of cats, where there are also a set of landmarks per photo.

Start by unpacking all the training photos in folder you choose.

## Preprocessing data for machine learning

The photos contains cats in many situations and backgrounds. To make it easier to do machine learning, we will *preprocess* the data, so the photos only contains the face of the cat. Preprocessing is and important step in most machine learning approaches.

The preprocessing steps are:

- Define a model cat (`ModelCat.jpg`) with associated landmarks (`ModelCat.jpg.cat`)
- For each cat in the training data:
  - Use landmark based registration with a *similarity transform* to register the photo to the model cat
  - Crop the registered photo
  - Save the result in a fold called **preprocessed**

**Exercise 1:** *Preprocess all image in the training set. To do the preprocessing, you can use the code snippets supplied* [here](https://github.com/RasmusRPaulsen/DTUImageAnalysis/tree/main/exercises/ex8-CatsCatsCats/data)

The result of the preprocessing is a directory containing smaller photos of the same shape containing cat faces.

## Gathering data into a data matrix

To start, we want to collect all the image data into a data matrix. The matrix should have dimensions `[n_samples, n_features]` where **n_samples** is the number of photos in our training set and **n_features** is the number of values per image. Since we are working with RGB images, the number of features are given by `n_features = height * width * 3`. 

The data matrix can be constructed by:

- Find the number of image files in the **preprocessed** folder using `glob`
- Read the first photo and use that to find the height and width of the photos
- Set **n_samples** and **n_features**
- Make an empty matrix `data_matrix = np.zeros((n_samples, n_features))`
- Read the image files one by one and use `flatten()` to make each image into a 1-D vector (flat_img). 
- Put the image vector (flat_img) into the data matrix by for example `data_matrix[idx, :] = flat_img` , where idx is the index of the current image.

**Exercise 2:** *Compute the data matrix.* 

## Compute and visualize a mean cat

In the data matrix, one row is one cat. You can therefore compute an average cat, **The Mean Cat** by computing one row, where each element is the average of the column. 

**Exercise 2:** *Compute the average cat.* 

You can use the supplied function `create_u_byte_image_from_vector` to create an image from a 1-D image vector.

**Exercise 3:** *Visualize the Mean Cat* 

## Find a missing cat or a cat that looks like it

Oh! no! You were in such a hurry to get to DTU that you forgot to close your window. Now your cat is gone!!! What to do? 

**Exercise 4:** *Decide that you quickly buy a new cat that looks very much like the missing cat - so nobody notices* 

To find a cat that looks like the missing cat, you start by comparing the missing cat pixels to the pixels of the cats in the training set. The comparison between missing cat data and the training data can be done using the sum-of-squared differences (SSD).

**Exercise 5:** *Use the `preprocess_one_cat` function to preprocess the photo of the poor missing cat*

**Exercise 6:** *Flatten the pixel values of the missing cat so it becomes a vector of values.*

**Exercise 7:** *Subtract you missing cat data from all the rows in the data_matrix and for each row compute the sum of squared differences. This can for example be done by `sub_distances = np.linalg.norm(sub_data, axis=1)`, where sub_data are the subtracted pixel data.*

**Exercise 8:** *Find the training cat that looks most like your missing cat by finding the cat, where the SSD is smallest. You can for example use `np.argmin`.

**Exercise 9:** *Extract the found cat from the data_matrix and use `create_u_byte_image_from_vector` to create an image that can be visualized. Did you find a good replacement cat?*


## References
- [Cat data set](https://www.kaggle.com/datasets/crawford/cat-dataset)
