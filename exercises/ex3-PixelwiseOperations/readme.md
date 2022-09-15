# Exercise3 - Pixelwise operations (WORK IN PROGRESS)

In this exercise you will learn to perform pixelwise operations using Python.


# Learning Objectives

After completing this exercise, the student should be able to do the following:

1. Implement and test a function that can do linear histogram stretching of a grey level image.
2. Convert image pixels between doubles and 8-bit unsigned integers (UINT8)
3. Implement and test a function that can perform gamma mapping of a grey level image.
4. Implement and test a function that can threshold a grey scale image.
5. Use Otsu's automatic method to compute an optimal threshold that seperates foreground and background
6. Perform RGB thresholding in a color image.
7. Convert a RGB image to HSV using the function `rgsb2hsv` from the `skimage.color` package.
8. Visualise individual H, S, V components of a color image.
9. Implement and test thresholding in HSV space.


# Installing Python packages

In this exercise, we will be using both [scikit-image](https://scikit-image.org/) and [OpenCV](https://opencv.org/). You should have both libraries installed, else instructions can be found in the previous exercises.

We will use the virtual environment from the previous exercise (`course02502`). 

# Exercise data and material

The data and material needed for this exercise can be found here:
(https://github.com/RasmusRPaulsen/DTUImageAnalysis/blob/main/exercises/ex3-PixelwiseOperations/data/)

# Explorative data analysis

First we will be working with an X-ray image of the human vertebra, `vertebra.png`. This type of images can for example be used for diagnosis of osteoporosis. A symptom is the so-called vertebral compression fracture. However, the diagnosis is very difficult to do based on x-rays alone.

### Exercise 1

Start by reading the image and inspect the histogram. Is it a *bimodal* histogram? Do you think it will be possible to segment it so only the bones are visible? 

### Exercise 2

Compute the minimum and maximum values of the image. Is the full scale of the gray-scale spectrum used or can we enhance the appearance of the image?

# Pixel type conversions

Before going further, we need to understand how to convert between between pixel types and what should be considered. A comphrehensive guide can be found [here](https://scikit-image.org/docs/stable/user_guide/data_types.html) (it is not mandatory reading, we just use some highlights). One important point is that we should avoid using the `astype` function on images. We did that in ex2b, but we will not so that anymore.

## Conversion from unsigned byte to float image

In *unsigned byte* images, the possible pixel value range is [0, 255]. When converting an *unsigned byte* image to a *float* image, the possible pixel value range will be [0, 1]. When you use Python skimage function `img_as_float` on an *unsigned byte* image, it will automatically divide all pixel values with 255.

### Exercise 3

Add an import statement to your script:
```
from skimage.util import img_as_float
from skimage.util import img_as_ubyte
```

Read the image `vertebra.png` and compute and show the minumum and maximum values.

Use `img_as_float` to compute a new float version of your input image. Compute the minimum and maximum values of this float image. Can you verify that the float image is equal to the original image, where each pixel value is divided by 255?

## Conversion from float image to unsigned byte image

As stated above, an (unsigned) float image can have pixel values in [0, 1]. When using the Python skimage function `img_as_ubyte` on an (unsigned) float image, it will multiply all values with 255 before converting into a byte. Remember that all decimal number will be converted into integers by this, and some information might be lost.

### Exercise 4

Use `img_as_ubyte` on the float image you computed in the previous exercise. Compute the Compute the minimum and maximum values of this image. Are they as expected?


# Histogram stretching

You should implement a function, that automatically stretches the histogram of an image. In other words, the function should create a new image, where the pixel values are changed so the histogram of the output image is *optimal*. Here *optimal* means, that the minimum value is 0 and the maximum value is 255. It should be based on the *linear histogram stretching* equation:

$$g(x,y) =\frac{v_\text{max,d}-v_\text{min,d}}{v_\text{max}-v_\text{min}}(f(x,y) - v_\text{min} )+v_\text{min,d} \enspace .$$

Here $f(x,y)$ is the input pixel value and  $g(x,y)$ is the output pixel value, $v_\text{max,d}$ and $v_\text{min,d}$ are the desired minimum and maximum values (0 and 255) and  $v_\text{max}$ and $v_\text{min}$ are the current minumum and maximum values.

### Exercise 5

Implement a Python function called `histogram_stretch`. It can, for example, follow this example:

```python
def histogram_stretch(img_in):
    """
    Stretches the histogram of an image 
    :param img_in: Input image
    :return: Image, where the histogram is stretched so the min values is 0 and the maximum value 255
    """
    # img_as_float will divide all pixel values with 255.0
    img_float = img_as_float(img_in)
    min_val = img_float.min()
    max_val = img_float.max()
    min_desired = 0.0
    max_desired = 1.0
	
    # Do something here

    # img_as_ubyte will multiply all pixel values with 255.0 before converting to unsigned byte
    return img_as_ubyte(img_out)
```

### Exercise 6

Test your `histogram_stretch` on the `vertebra.png` image. Show the image before and after the histogram stretching. What changes do you notice in the image? Are the important structures more visible?


# Non-linear pixel value mapping

The goal is to implement and test a function that performs a $\gamma$-mapping of pixel values:

$$g(x,y) = f(x,y)^\gamma \enspace .$$

You can use the *Numpy* function `power` to compute the actual mapping function. 

### Exercise 7

Implement a function, `gamma_map(img, gamma)`, that:

1. Converts the input image to float
2. Do the gamma mapping on the pixel values
3. Returns the resulting image as an unsigned byte image.

### Exercise 8

Test your `gamma_map` function on the vertebra image or another image of your choice. Try different values of $\gamma$, for example 0.5 and 2.0. Show the resuling image together with the input image. Can you see the differences in the images?


# Real time pixelwise operations on videos

In the [exercise material](https://github.com/RasmusRPaulsen/DTUImageAnalysis/blob/main/exercises/ex3-PixelwiseOperations/data/), there is a Python script using OpenCV that:

1. Connects to a camera
2. Acquire images, converts them to gray-scale and after that to floating point images
3. Do a simple processing on the gray-scale (inversion) or the colour image (inversion of the red channel)
4. Computes the frames per second (fps) and shows it on an image.
5. Shows input and resulting images in windows.
6. Checks if the key `q` has been pressed and stops the program if it is pressed.

It is possible to use a mobile phone as a remote camera by following the instructions in exercise 2b.

### Exercise 9

Run the program from the [exercise material](https://github.com/RasmusRPaulsen/DTUImageAnalysis/blob/main/exercises/ex3-PixelwiseOperations/data/) and see if it shows the expected results? 

### Exercise 10

Change the gray-scale processing in the [exercise material](https://github.com/RasmusRPaulsen/DTUImageAnalysis/blob/main/exercises/ex3-PixelwiseOperations/data/) script to be for example thresholding, gamma mapping or something else. Do you get the visual result that you expected?


### Exercise 11: Real time detection of DTU signs

Change the gray-scale processing in the [exercise material](https://github.com/RasmusRPaulsen/DTUImageAnalysis/blob/main/exercises/ex3-PixelwiseOperations/data/) script so it does a color threshold in either RGB or HSV space. The goal is to make a program that can *see* DTU street signs. The output should be a binary image, where the pixels of the sign is foreground. Later in the course, we will learn how to remove the noise pixels.
