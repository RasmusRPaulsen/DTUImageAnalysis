# Exercise 8 - Cats, Cats, Cats

Are you sad that you have watched all cat movies and seen all cat photos on the internet? Then be sad no more - in this exercise we will make a *Cat Synthesizer* where you can create all the cat photos you will ever need!

Also, you can help your friends with missing cats to find the perfect new *twin cat*.
 
To be able to do these wonderful things we will harness the power of image based *principal component analysis*. The methods we will use, can be called *classical machine learning*.


## Learning Objectives

After completing this exercise, the student should be able to do the following:

1. compute PCA on images

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

The data and material needed for this exercise can be found here: [exercise data and material]
(https://github.com/RasmusRPaulsen/DTUImageAnalysis/tree/main/exercises/ex8-CatsCatsCats/data)

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

**Exercise 1:** *Preprocess all image in the training set. To do the preprocessing, you can use the code snippets supplied [here]
(https://github.com/RasmusRPaulsen/DTUImageAnalysis/tree/main/exercises/ex8-CatsCatsCats/data)*

The result of the preprocessing is a directory containing smaller photos of the same shape containing cat faces.


## References
- [Cat data set](https://www.kaggle.com/datasets/crawford/cat-dataset)
