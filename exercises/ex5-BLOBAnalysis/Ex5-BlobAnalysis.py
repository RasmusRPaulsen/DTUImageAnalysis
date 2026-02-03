from skimage import io, color, morphology
from skimage.util import img_as_float, img_as_ubyte
import matplotlib.pyplot as plt
import numpy as np
import math
from skimage.filters import threshold_otsu
from skimage import segmentation
from skimage import measure
from skimage.color import label2rgb


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


# https://scikit-image.org/docs/0.19.x/auto_examples/segmentation/plot_label.html#sphx-glr-auto-examples-segmentation-plot-label-py
def blob_analysis():
    in_dir = "data_ex5/"
    im_name = "lego_4_small.png"
    im_org = io.imread(in_dir + im_name)
    # io.imshow(im_org)
    # io.show()

    im_gray = color.rgb2gray(im_org)
    thres = threshold_otsu(im_gray)
    bin_img = im_gray < thres
    # io.imshow(img_as_ubyte(bin_img))
    # plt.title('Binary edges. Otsu')
    # io.show()
    # show_comparison(im_org, bin_img, 'Binary image')

    img_c_b = segmentation.clear_border(bin_img)

    se = morphology.disk(5)
    img_closed = morphology.binary_closing(img_c_b, se)
    img_open = morphology.binary_opening(img_closed, se)

    label_img = measure.label(img_open)
    n_labels = label_img.max()
    print(f"Number of labels: {n_labels}")

    # to make the background transparent, pass the value of `bg_label`,
    # and leave `bg_color` as `None` and `kind` as `overlay`
    # image_label_overlay = label2rgb(label_img, image=im_org, bg_label=0)
    image_label_overlay = label2rgb(label_img)
    show_comparison(im_org, image_label_overlay, 'BLOBS')

    # io.imshow(image_label_overlay)
    # io.show()
    region_props = measure.regionprops(label_img)
    # print(region_props)

    areas = np.array([prop.area for prop in region_props])
    print(areas)
    plt.hist(areas, bins=50)
    plt.show()


def cell_counting():
    in_dir = "data/"
    img_org = io.imread(in_dir + 'Sample E2 - U2OS DAPI channel.tiff')
    # slice to extract smaller image
    img_small = img_org[700:1200, 900:1400]
    img_gray = img_as_ubyte(img_small)
    io.imshow(img_gray, vmin=0, vmax=150)
    plt.title('DAPI Stained U2OS cell nuclei')
    io.show()

    # avoid bin with value 0 due to the very large number of background pixels
    plt.hist(img_gray.ravel(), bins=256, range=(1, 100))
    io.show()

    threshold = 20
    img_bin = img_gray > threshold
    show_comparison(img_small, img_bin, "Binary image")
    img_c_b = segmentation.clear_border(img_bin)
    label_img = measure.label(img_c_b)
    image_label_overlay = label2rgb(label_img)
    show_comparison(img_org, image_label_overlay, 'Found BLOBS')

    # io.imshow(L)
    # io.show()

    region_props = measure.regionprops(label_img)
        # print(RP[0].area)

    areas = np.array([prop.area for prop in region_props])

    plt.hist(areas)
    plt.show()

    # areas.sort()

    min_area = 10
    max_area = 150

    # Create a copy of the label_img
    label_img_filter = label_img
    for region in region_props:
        # Find the areas that do not fit our criteria
        if region.area > max_area or region.area < min_area:
            # set the pixels in the invalid areas to background
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0
    # Create binary image from the filtered label image
    i_area = label_img_filter > 0
    show_comparison(img_small, i_area, 'Found nuclei based on area')

    perimeters = np.array([prop.perimeter for prop in region_props])
    plt.plot(areas, perimeters, '.b')
    plt.show()

    circularity = 4 * math.pi * areas / (perimeters * perimeters)
    plt.hist(circularity, bins=100)
    plt.title("Circularity")
    plt.show()

    plt.plot(areas, circularity, '.b')
    plt.title("Area vs circularity")
    plt.show()

    min_area = 10
    max_area = 150
    label_img_filter = label_img
    num_objects = 0
    for region in region_props:
        p = region.perimeter
        a = region.area
        circularity = 4 * math.pi * a / (p * p)
        if circularity < 0.7 or a < min_area or a > max_area:
            for cords in region.coords:
                label_img_filter[cords[0], cords[1]] = 0
        else:
            num_objects = num_objects + 1

    print("Objects: ", num_objects)
    i_circ = label_img_filter > 0
    show_comparison(img_small, i_circ, 'Found nuclei based on area and circularity')


if __name__ == '__main__':
    blob_analysis()
    # cell_counting()

