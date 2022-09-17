# Exercise 4 - Image Filtering (WORK IN PROGRESS)

The purpose of this exercise is to illustrate different image filtering techniques.


# Learning Objectives

After completing this exercise, the student should be able to do the following:

1. Filtering images

# Installing Python packages

In this exercise, we will be using both [scikit-image](https://scikit-image.org/), [OpenCV](https://opencv.org/) and [SciPy](https://scipy.org/). You should have these libraries installed, else instructions can be found in the previous exercises.

We will use the virtual environment from the previous exercise (`course02502`). 

# Exercise data and material

The data and material needed for this exercise can be found here:
(https://github.com/RasmusRPaulsen/DTUImageAnalysis/blob/main/exercises/ex4-ImageFiltering/data/)

# Filtering using Python

scikit-image and SciPy contain a large number of image filtering functions. In this exercise, we will explore some of the fundamental functions and touch upon more advanced filters as well.

## Filtering using correlation

We will start by exploring the basic correlation operator from SciPy. Start by importing:

```python
from scipy.ndimage import correlate
```

Now create a small and simple image:

```python
input_img = np.arange(25).reshape(5, 5)
print(input_img)
```

and a simple filter:
```python
weights = [[0, 1, 0],
		   [1, 2, 1],
		   [0, 1, 0]]

```

Now we can correlate the image with the weights:

```python
res_img = correlate(input_img, weights)
```

### Exercise 1

Print the value in position (3, 3) in `res_img`. Explain the value?

## Border handling 

We can choose border of the image is handled when doing the correlation. By default, correlation uses *reflection*. We can set it to have a constant value (for example 10) outside the image by:

```python
res_img = correlate(input_img, weights, mode="constant", cval=10)
```

### Exercise 2

Compare the output images when using `reflection` and `constant` for the border. Where and why do you see the differences.

## Mean filtering

Now we will try some filters on an artificial image with different types of noise. 

### Exercise 3

Read and show the image **Gaussian.png** from the [exercise material](https://github.com/RasmusRPaulsen/DTUImageAnalysis/blob/main/exercises/ex4-ImageFiltering/data/).

Create a mean filter with normalized weights:
```python
size = 5
# Two dimensional filter filled with 1
weights = np.ones([size, size])
# Normalize weights
weights = weights / np.sum(weights)
```

Use `correlate` with the **Gaussian.png** image and the mean filter. Show the resulting image together with the input image. What do you observe?

Try to change the size of the filter to 10, 20, 40 etc.. What do you see?



# Explorative data analysis



## References
[sci-kit image filters](https://scikit-image.org/docs/stable/api/skimage.filters.html)
[rank filters](https://scikit-image.org/docs/stable/auto_examples/applications/plot_rank_filters.html)
[scipy correlate](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.correlate.html)
