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

Implement and test a Python function called `histogram_stretch`. It can, for example, follow this example:

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

    return img_as_ubyte(img_out)
```




# OpenCV program for image differencing

In the [exercise material](https://github.com/RasmusRPaulsen/DTUImageAnalysis/blob/main/exercises/ex2b-ChangeDetectionInVideos/data/), there is a Python script using OpenCV that:

1. Connects to a camera
2. Acquire images, converts them to gray-scale and after that to floating point images
3. Computes a difference image between a current image and the previous image.
4. Computes the frames per second (fps) and shows it on an image.
5. Shows images in windows.
6. Checks if the key `q` has been pressed and stops the program if it is pressed.

It is possible to use a mobile phone as a remote camera by following the instructions in [Using a mobile phone](#using-a-mobile-phone-camera).

Note that we sometimes refers to an image as a *frame*.

### Exercise 1

Run the program from the [exercise material](https://github.com/RasmusRPaulsen/DTUImageAnalysis/blob/main/exercises/ex2b-ChangeDetectionInVideos/data/) and see if shows the expected results? Try to move your hands in front of the camera and try to move the camera and see the effects on the difference image.

### Exercise 2

Identify the important steps above in the program. What function is used to convert a color image to a gray-scale image?

# Change detection by background subtraction

The goal of this exercise, is to modify the program in the [exercise material](https://github.com/RasmusRPaulsen/DTUImageAnalysis/blob/main/exercises/ex2b-ChangeDetectionInVideos/data/), so it will be able to raise an alarm if significant changes are detected in a video stream.

The overall structure of the program should be:



- Connect to camera
- Acquire a background image, convert it to grayscale and then to floating point
- Start a loop:
	1. Acquire a new image, convert it to grayscale and then to floating point: $I_\text{new}$ .
    2. Computes an absolute difference image between the new image and the background image.
    3. Creates a binary image by applying a threshold, T, to the difference image.
    4. Computes the total number of foreground, F, pixels in the foreground image.
	5. Compute the percentage of foreground pixels compared to the total number of pixels in the image (F).
    5. Decides if an alarm should be raised if F is larger than an alert value, A.
    6. If an alarm is raised, show a text on the input image. For example **Change Detected!**.
    7. Shows the input image, the backround image, the difference image, and the binary image. The binary image should be scaled by 255.
    8. Updates the background image, $I_\text{background}$, using: $$I_\text{background} = \alpha * I_\text{background} + (1 - \alpha) * I_\text{new}$$
    9. Stop the loop if the key `q` is pressed.

You can start by trying with $\alpha = 0.95$, $T = 10$, and $A = 0.05$.

### Exercise 3

Implement and test the above program.

### Exercise 4

Try to change $\alpha$, $T$ and $A$. What effects do it have?

### Exercise 5

The images are displayed using the OpenCV function `imshow`. The display window has several ways of zooming in the displayed image. One function is to zoom x30 that shows the pixel values as numbers. Do that and notice the change of the values.

### Exercise 6

Try to use `putText` to write some important information on the image. For example the number of changed pixel, the average, minumum and maximum value in the difference image. These values can then be used to find even better values for $\alpha$, $T$ and $A$.

Also try to find out how to put a colored text on a color image. Here you need to know that OpenCV stores color as BGR instead of RGB.

# Using a mobile phone camera

It is possible to use a mobile phone as a remote camera in OpenCV.

You need to install a web cam app on your phone. One option is `DroidCam` that can be installed from Google Play or from Apple App Store.

The computer and your phone should be on the same wireless network. For example one of the DTU wireless networks.

Now start the DroidCam application on your phone. It should now show an web-address, for example `http://192.168.1.120:4747/video`

Use this address, in the program:

```python
use_droid_cam = True
if use_droid_cam:
    url = "http://192.168.1.120:4747/video"
cap = cv2.VideoCapture(url)
```

You should now see the video from your mobile phone on your computer screen. Remember you phone should be unlocked when streaming video.


