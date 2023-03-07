# Modified sligthly from
# https://scikit-image.org/docs/dev/auto_examples/segmentation/plot_regionprops.html
import plotly
import plotly.express as px
import plotly.graph_objects as go
from skimage import data, filters, measure, morphology, io, color

# spyder fix
import plotly.io as pio 
pio.renderers.default = 'browser'

def interactive_blobs():
    in_dir = "data/"
    im_name = "lego_4_small.png"

    img_org = io.imread(in_dir + im_name)
    img = color.rgb2gray(img_org)
    # Binary image, post-process the binary mask and compute labels
    threshold = filters.threshold_otsu(img)
    mask = img < threshold
    mask = morphology.remove_small_objects(mask, 50)
    mask = morphology.remove_small_holes(mask, 50)
    labels = measure.label(mask)

    fig = px.imshow(img, binary_string=True)
    fig.update_traces(hoverinfo='skip') # hover is only for label info

    props = measure.regionprops(labels, img)
    properties = ['area', 'eccentricity', 'perimeter', 'intensity_mean']

    # For each label, add a filled scatter trace for its contour,
    # and display the properties of the label in the hover of this trace.
    for index in range(1, labels.max()):
        label_i = props[index].label
        contour = measure.find_contours(labels == label_i, 0.5)[0]
        y, x = contour.T
        hoverinfo = ''
        for prop_name in properties:
            hoverinfo += f'<b>{prop_name}: {getattr(props[index], prop_name):.2f}</b><br>'
        fig.add_trace(go.Scatter(
            x=x, y=y, name=label_i,
            mode='lines', fill='toself', showlegend=False,
            hovertemplate=hoverinfo, hoveron='points+fills'))

    plotly.io.show(fig)


if __name__ == '__main__':
    interactive_blobs()

