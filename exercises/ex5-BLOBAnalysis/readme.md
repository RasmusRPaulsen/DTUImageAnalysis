# Exercise 5 - BLOB Analysis (connected component analysis and object classification) (WORK IN PROGRESS)

The purpose of this exercise is to implement, test and validate connected component analysis methods. Also known as BLOB (binary large object) analysis.

# Learning Objectives

After completing this exercise, the student should be able to do the following:

1. Do BLOB analysis

# Installing Python packages

In this exercise, we will be using [scikit-image](https://scikit-image.org/). You should have this library installed, else instructions can be found in the previous exercises.

We will use the virtual environment from the previous exercise (`course02502`). 

# Exercise data and material

The data and material needed for this exercise can be found here:
(https://github.com/RasmusRPaulsen/DTUImageAnalysis/tree/main/exercises/ex5-BLOBAnalysis/data)

# BLOB Analysis in Python 

Start by importing some function:

```python
from skimage import io, color, morphology
from skimage.util import img_as_float, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.filters import threshold_otsu
from skimage import segmentation
from skimage import measure
from skimage.color import label2rgb
```

and define a convenience function to show two images side by side:

```python
def show_comparison(original, modified, modified_name):
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(8, 4), sharex=True,
                                   sharey=True)
    ax1.imshow(original)
    ax1.set_title('Original')
    ax1.axis('off')
    ax2.imshow(modified)
    ax2.set_title(modified_name)
    ax2.axis('off')
    io.show()
```


## Cell counting

The images used for the exercise is acquired by the Danish company [Chemometec](https://chemometec.com/) using their image-based cytometers. A cytometer is a machine used in many laboratories to do automated cell counting and analysis. An example image can be seen in below where U2OS cells  (human bone cells) have been imaged using ultraviolet (UV) microscopy and a fluorescent staining method named DAPI. Using DAPI staining only the cell nuclei are visible which makes the method very suitable for cell counting.

![UV](figs/U2OS-AP.png)
![DAPI](figs/U2US-DAPI.png)

The raw images from the Cytometer are 1920x1440 pixels and each pixel is 16 bit (values from 0 to 65535). The resolution is 1.11$\mu m$ / pixel.

To make it easier to develop the cell counting program we start by working with smaller areas of the raw images. The images are also converted to 8 bit grayscale images:

```python
I = io.imread('data/CellData/Sample E2 - U2OS DAPI channel.tiff')
J = I[700:1200,900:1400] # Slice to extract part of the image
G = img_as_ubyte(J)  # Convert to 8-bit grayscale
io.imshow(G, vmin=0, vmax=150)
plt.title('DAPI Stained U2OS cell nuclei')
io.show()
```

As can be seen we use \emph{slicing to extract a part of the image}. You can use \verb|vmin| and \verb|vmax|to visualise specific gray scale ranges (0 to 150 in the example above). Adjust these limits to find out where the cell nuclei are most visible.

Initially, we would like to apply a threshold to create a binary image where nuclei are foreground. To select a good threshold, inspect the histogram:

```python
plt.clf() # magic plot command
plt.cla() # magic plot command
plt.close() # magic plot command

# avoid bin with value 0 due to the very large number of background pixels
plt.hist(G.ravel(), bins=256, range =(1, 100))
io.show()
```

The \emph{magic plot commands} has been used to avoid weird plotting behaviour. Use them when you encounter problems!

Now select an appropriate threshold and apply it to the image:

```python
threshold =
BW = G > threshold
io.imshow(BW)
io.show()
```

It can be seen that there is some noise (non-nuclei) present and that some nuclei are connected. Nuclei that are overlapping very much should be discarded in the analysis. However, if they are only touching each other a little we can try to separate them. More on this later.

To make the following analysis easier the objects that touches the border should be removed.

```python
C = segmentation.clear_border(BW)
```

To be able to analyse the individual objects, the objects should be
labelled.

```python
L = measure.label(C)
io.imshow(L)
io.show()
```

In this image, each object has a separate color - does it look reasonable?

The task is now to find some {\em object features} that identifies the cell nuclei and let us remove noise and connected nuclei. We use the function \verb|regionprops| to compute a set of features for each object:

```python
RP = measure.regionprops(L)
```

For example can the area of the first object be found as \verb|print(RP[0].area)|.

A quick way to gather all areas:
```python
areas = np.array([prop.area for prop in RP])
```

We can see if the area of the objects is enough to remove invalid object. Plot a histogram of all the areas and see if it can be used to identify well separated nuclei from overlapping nuclei and noise. Use \verb|plt.hist()|.

Select a minimum and maximum allowed area and use the following to visualise the result:

```python
min_area =
max_area =

L2 = L
for region in RP:
    if region.area > max_area or region.area < min_area:
        for cords in region.coords:
            L2[cords[0], cords[1]] = 0 # blank the label
I_area = L2 > 0
io.imshow(I_area)
io.show()
```

Can you find an interval that works well for these nuclei?


We should also examine if the shape of the cells can identify them. A good measure of how circular an object is can be computed as:
\begin{equation}
f_\text{circ} = \frac{4 \pi A}{P^2},
\end{equation}

where $A$ is the object area and $P$ is the perimeter. A circle has a circularity close to 1, and very-non-circular object have circularity close to 0.

We start by getting all the object perimeters:

```python
perimeters = np.array([prop.perimeter for prop in RP])
```

\begin{ex}
Compute the circularity for all objects and plot a histogram.
\end{ex}

Select some appropriate ranges of accepted circularity. Use these ranges to select only the cells with acceptable areas and circularity and show them in an image.

\begin{ex}
Extend your method to return the number (the count) of well-formed nuclei in the image.
\end{ex}

Try to test the method on a larger set of training
  images. In the table below the suggested training images and regions are seen. Use slicing to select the correct regions from the raw image. You can also try it on regions that you select yourself.

\begin{tabular}{|l|r|r|}
  \hline
  File & Cell Type & Selection (x,y,width,heigh)\\
  \hline
  % after \\: \hline or \cline{col1-col2} \cline{col3-col4} ...
  Sample E2 - U2OS DAPI channel.tiff & U2OS & 700, 900, 500, 500 \\
  Sample E2 - U2OS DAPI channel.tiff & U2OS & 0, 0, 500, 500 \\
  Sample E2 - U2OS DAPI channel.tiff & U2OS & 600, 200, 500, 500 \\
  Sample E2 - U2OS DAPI channel.tiff & U2OS & 1300, 0, 500, 500 \\
  Sample E2 - U2OS DAPI channel.tiff & U2OS & 900, 500, 500, 500 \\
  Sample G1 - COS7 cells DAPI channel & COS7 & 0, 0, 500, 500 \\
  Sample G1 - COS7 cells DAPI channel & COS7 & 500, 0, 500, 500 \\
  Sample G1 - COS7 cells DAPI channel & COS7 & 0, 500, 500, 500 \\
  Sample G1 - COS7 cells DAPI channel & COS7 & 500, 500, 500, 500 \\
  Sample G1 - COS7 cells DAPI channel & COS7 & 1000, 0, 500, 500 \\
  Sample G1 - COS7 cells DAPI channel & COS7 & 1000, 500, 500, 500 \\
  \hline
\end{tabular}

COS7 cells are {\em African Green Monkey Fibroblast-like Kidney Cells} (www.cos-7.com) used for a variety of research purposes.


\subsection*{Handling overlap}

In certain cases cell nuclei are touching and are therefore being treated as one object. It can sometimes be solved using for example the morphological operation opening before the object labelling. The operation erosion can also be used but it changes the object area.

\begin{ex}
Experiment with morphological operations to see if you can separate touching cells before counting them.
\end{ex}




## Image morphology on a single object

An image, **lego_5.png** of a lego brick can be used to test some of the basic functions. 

### Exercise 1

We will start by computing a binary image from the lego image:

- Read the image into **im_org**.
- Convert the image to gray scale. 
- Find a threshold using *Otsu's method*.
- Apply the treshold and generate a binary image **bin_img**.
- Visualize the image using `plot_comparison(im_org, bin_img, 'Binary image')`

As ncan be seen, the lego brick is not *segmented* perfectly. There are holes in the segmentation. Let us see if what we can do.

### Exercise 2

We will start by creating a *structuring element*. In scikit-image they are called *footprint*. A disk shaped footprint can be created by:

```python
footprint = disk(2)
# Check the size and shape of the structuring element
print(footprint)
```

The morphological operation **erosion** can remove small objects, seperate objects and make objects smaller. Try it on the binary lego image:

```python
eroded = erosion(bin_img, footprint)
plot_comparison(bin_img, eroded, 'erosion')
```

Experiement with different sizes of the footprint and observe the results.


### Exercise 3

The morphological operation **dilation** makes objects larger, closes holes and connects objects. Try it on the binary lego image:

```python
dilated = dilation(bin_img, footprint)
plot_comparison(bin_img, dilated, 'dilation')
```

Experiement with different sizes of the footprint and observe the results.


### Exercise 4

The morphological operation **opening** removes small objects without changing the size of the remaining objects. Try it on the binary lego image:

```python
opened = opening(bin_img, footprint)
plot_comparison(bin_img, opened, 'opening')
```

Experiement with different sizes of the footprint and observe the results.

### Exercise 5

The morphological operation **closing** closes holes in objects without changing the size of the remaining objects. Try it on the binary lego image:

```python
closed = closing(bin_img, footprint)
plot_comparison(bin_img, closed, 'closing')
```

Experiement with different sizes of the footprint and observe the results.

## Object outline

It can be useful to compute the outline of an object both to measure the perimeter but also to see if it contains holes or other types of noise. Start by defining an outline function:

```python
def compute_outline(bin_img):
    """
    Computes the outline of a binary image
    """
    footprint = disk(1)
    dilated = dilation(bin_img, footprint)
    outline = np.logical_xor(dilated, bin_img)
    return outline
```

### Exercise 6

Compute the outline of the binary image of the lego brick. What do you observe?

### Exercise 7

Try the following:

- Do an *opening* with a disk of size 1 on the binary lego image.
- Do a *closing* with a disk of size 15 on the result of the opening.
- Compute the outline and visualize it.

What do you observe and why does the result look like that?

## Morphology on multiple objects

Let us try to do some analysis on images with multiple objects.

### Exercise 8

Start by:
- reading the **lego_7.png** image and convert it to gray scale.
- Compute a treshold using *Otsu's method* and apply it to the image.
- Show the binary image together with the original.
- Compute the outline of the binary image and show it with the binary image.

What do you observe?

### Exercise 9

We would like to find a way so only the outline of the entire brick is computed. So for each lego brick there should only be one closed curve.

Try using the *closing* operations and find out which size of footprint that gives the desired result?

### Exercise 10

Try the above on the **lego_3.png** image. What do you observe?
 

## Morphology on multiple connected objects

Morphology is a strong tool that can be used to clean images and seperate connected objects. In image **lego_9.png** some lego bricks are touching. We would like to see if we can seperate them.

### Exercise 11

Start by:
- reading the **lego_9.png** image and convert it to gray scale.
- Compute a treshold using *Otsu's method* and apply it to the image.
- Show the binary image together with the original.
- Compute the outline of the binary image and show it with the binary image.

What do you observe?

### Exercise 12

Let us start by trying to remove the noise holes inside the lego bricks. Do that with an *closing* and find a good footprint size. Compute the outline and see what you observe?

### Exercise 13

Now we will try to seperate the objects. Try using a *erosion* on the image that you repaired in exercise 12. You should probably use a rather large footprint. How large does it need to be in order to split the objects?

### Exercise 14

The objects lost a lot of size in the previous step. Try to use *dilate* to make them larger. How large can you make them before they start touching?

## Puzzle piece analysis

We would like to make a program that can help solving puzzles. The first task is to outline each piece. A photo, **puzzle_pieces.png** is provided. 

### Exercise 15

Use the previosly used methods to compute a binary image from the puzzle photo. What do you observe?

### Exercise 16

Try to use a an *opening* with a large footprint to clean the binary. Compute the outline. Do we have good outlines for all the pieces?


The conclusion is that you can solve a lot of problems using morphological operations but sometimes it is better to think even more about how to acquire the images.


## References
- [sci-kit image region properties](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops)
