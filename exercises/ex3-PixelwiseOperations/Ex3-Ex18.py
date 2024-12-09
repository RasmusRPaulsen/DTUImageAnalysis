from skimage import color
from skimage.filters import threshold_otsu
from skimage.util import img_as_ubyte
from skimage.util import img_as_float
import numpy as np
import time
import cv2


def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name, img)


def process_hsv_image(img):
    """
    Simple processing of a color (HSV) image
    """
    hue_img = img[:, :, 0]    
    segm_red = (hue_img < 1.0) & (hue_img  > 0.9)
    return img_as_ubyte(segm_red)

def process_rgb_image(img):
    """
    Segmentation of red structures in the RGB channel
    """
    r_comp = img[:, :, 0]
    g_comp = img[:, :, 1]
    b_comp = img[:, :, 2]
    
    segm = (r_comp > 160) & (r_comp < 180) & (g_comp > 50) & (g_comp < 80) & \
                (b_comp > 50) & (b_comp < 80)

    return img_as_ubyte(segm)


def capture_from_camera_and_show_images():
    print("Starting image capture")

    print("Opening connection to camera")
    url = 0
    use_droid_cam = False
    if use_droid_cam:
        url = "http://192.168.1.120:4747/video"
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    print("Starting camera loop")
    # To keep track of frames per second using a high-performance counter
    old_time = time.perf_counter()
    fps = 0
    stop = False
    process_rgb = False
    while not stop:
        ret, new_frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Change from OpenCV BGR to scikit image RGB
        new_image = new_frame[:, :, ::-1]
        if process_rgb:
            mask_red = process_rgb_image(new_image)
        else:
            new_image = color.rgb2hsv(new_image)
            mask_red = process_hsv_image(new_image)

        # update FPS - but do it slowly to avoid fast changing number
        new_time = time.perf_counter()
        time_dif = new_time - old_time
        old_time = new_time
        fps = fps * 0.95 + 0.05 * 1 / time_dif

        # Put the FPS on the new_frame
        str_out = f"fps: {int(fps)}"
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(new_frame, str_out, (100, 100), font, 1, 255, 1)

        # Display the resulting frame
        show_in_moved_window('Input', new_frame, 0, 10)
        show_in_moved_window('Mask', mask_red, 600, 10)
        if cv2.waitKey(1) == ord('q'):
            stop = True

    print("Stopping image loop")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    capture_from_camera_and_show_images()
