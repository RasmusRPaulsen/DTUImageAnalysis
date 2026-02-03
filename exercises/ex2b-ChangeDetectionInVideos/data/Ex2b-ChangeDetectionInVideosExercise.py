import time
import cv2
import numpy as np
from skimage.util import img_as_float
from skimage.util import img_as_ubyte


# esto es solo para mostrar la ventana luego con un nombre 
def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    cv2.namedWindow(win_name)
    #mueve la pantalla de posición
    cv2.moveWindow(win_name, x, y)
    # la muestra 
    cv2.imshow(win_name,img)


def capture_from_camera_and_show_images():
    print("Starting image capture")
    print("Opening connection to camera")
    url = 0 #esto es solo si quieres meterle un dispositivo nuevo
    use_droid_cam = False
    if use_droid_cam:
        url = "http://192.168.1.120:4747/video"
            #si encuentra un nuevo dispositivo url cambia, si es 0 , toma la del dispositivo

    # para aabrir la camara
    cap = cv2.VideoCapture(url)
    # cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Cannot open camera")
        exit() # se termina

    print("Starting camera loop")
    # Get first image
    ret, frame = cap.read() # ret es un booleano para ver si ha pillado el primer fotograma o no
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame")
        exit()

    # Transform image to gray scale and then to float, so we can do some processing
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = img_as_float(frame_gray)

    # To keep track of frames per second
    start_time = time.time() # guarda el tiempo cuando empieza a capturar
    n_frames = 0
    stop = False
    while not stop:
        ret, new_frame = cap.read()
        if not ret:
            print("Can't receive frame. Exiting ...")
            break

        # Transform image to gray scale and then to float, so we can do some processing
        new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY)
        new_frame_gray = img_as_float(new_frame_gray)

        # Compute difference image
        dif_img = np.abs(new_frame_gray - frame_gray)

        # Keep track of frames-per-second (FPS)
        #dividiendo el número de fotogramas procesados entre el tiempo transcurrido desde 
        n_frames = n_frames + 1
        elapsed_time = time.time() - start_time
        fps = int(n_frames / elapsed_time)

        # Put the FPS on the new_frame
        str_out = f"fps: {fps}"
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(new_frame, str_out , (100, 100), font, 1, 255, 1)

        # Display the resulting frame
        show_in_moved_window('Input', new_frame, 0, 10)
        show_in_moved_window('Input gray', new_frame_gray, 600, 10)
        show_in_moved_window('Difference image', dif_img, 1000, 10) # era 1200, 10  en teroria pero se te salia del ordenador

        # Old frame is updated
        frame_gray = new_frame_gray
        # si aprietas la q la cámara se cierra
        if cv2.waitKey(1) == ord('q'):
            stop = True

    print("Stopping image loop")
    cap.release() # libera la camara 
    cv2.destroyAllWindows() # cierra las ventanas

# paa qeu solo lo puedas llamar ejecutando este script, pero no con modulos y llmando a la funciones desde otro scypt
if __name__ == '__main__':
    capture_from_camera_and_show_images()
