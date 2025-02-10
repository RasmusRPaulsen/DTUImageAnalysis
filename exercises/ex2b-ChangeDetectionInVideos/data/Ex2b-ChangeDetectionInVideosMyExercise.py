import sys
import time
import cv2
import numpy as np


def show_in_moved_window(win_name, img, x, y):
    """
    Show an image in a window, where the position of the window can be given
    """
    # img = cv2.resize(img, (img.shape[1] * 30, img.shape[0] * 30), interpolation=cv2.INTER_NEAREST)
    cv2.namedWindow(win_name)
    cv2.moveWindow(win_name, x, y)
    cv2.imshow(win_name,img)


def capture_from_camera_and_show_images(alpha = 0.95, T = 10, A = 15000, exercise6 = False):
    print("Starting image capture")

    print("Opening connection to camera")
    url = 0  #esto es solo si quieres meterle un dispositivo nuevo
    use_droid_cam = False
    if use_droid_cam:
        url = "http://192.168.1.120:4747/video"
    #si encuentra un nuevo dispositivo url cambia, si es 0 , toma la del dispositivo

    # para aabrir la camara
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        print("Cannot open camera")
        exit() # se termina

    print("Starting camera loop")
    # Get first image
    ret, frame = cap.read() # ret es un booleano para ver si ha pillado el primer fotograma o no
    print(ret, frame)
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame")
        exit()


    # Transform image to gray scale and then to float, so we can do some processing
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    import matplotlib.pyplot as plt
    #force to show the image
    
    

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
        new_frame_gray = cv2.cvtColor(new_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

        # Compute difference image

        #Cada píxel tiene un valor numérico que representa la magnitud de la diferencia, no simplemente si existe o no un cambio.
        dif_img = np.abs(new_frame_gray - frame_gray)

        # NEW: Apply a threshold (filtra ruido y cambios importantes)
        # un valor umbral de 10. Los píxeles que tienen una diferencia mayor que 
        # ese 10 se ponen a TRUE y si es menor  a False.
        bin_img = dif_img > T 

        # NEW: Compute the number of px in the foreground
        # suma todos los 1 o trues que haya en la imagen binaria
        n_px_foreground = np.sum(bin_img)

        # NEW: Alarm A es otro
        if n_px_foreground > A:
            font = cv2.FONT_HERSHEY_COMPLEX
            cv2.putText(new_frame, "ALERT", (10, 50), font, 1, [0, 0, 255], 1)

        # Keep track of frames-per-second (FPS)
        n_frames = n_frames + 1
        elapsed_time = time.time() - start_time
        fps = int(n_frames / elapsed_time)

        # Put the FPS on the new_frame
        str_out = f"fps: {fps}"
        font = cv2.FONT_HERSHEY_COMPLEX
        cv2.putText(new_frame, str_out, (100, 100), font, 1, 255, 1)

        # NEW: Extra info   
        #Ejercicio 6: For example the number of changed pixel, the average, minumum and maximum value in the difference image. 
        #These values can then be used to find even better values for $\alpha$, $T$ and $A$.
        if exercise6:
            str_out = f"Changed px: {n_px_foreground}"
            cv2.putText(new_frame, str_out, (10, 470), font, 1, [255, 0, 255], 1) # Purple

            # Others:
            # Average value in diff image: np.mean(dif_img)
            # Max value in dif image: np.max(dif_img)
            # Min value in dif image: np.min(dif_img)

        # Display the resulting frame
        show_in_moved_window('Input', new_frame, 0, 10)
        show_in_moved_window('Difference image', dif_img.astype(np.uint8), 600, 10)
        show_in_moved_window('Binary image', (bin_img*255).astype(np.uint8), 1200, 10)  # al multiplicar por 255 todos los unos seran blancos

        # New: Old frame is updated
        #Su propósito es mantener una referencia del fondo que se adapte gradualmente a los cambios en la escena.
        frame_gray = alpha*frame_gray+(1-alpha)*new_frame_gray

        if cv2.waitKey(1) == ord('q'):
            stop = True

    print("Stopping image loop")
    cap.release()
    cv2.destroyAllWindows()
    
    fig, ax = plt.subplots()
    img_plot = ax.imshow(frame_gray, cmap='gray')

    # Display pixel values on the image
    for i in range(frame_gray.shape[0]):
        for j in range(frame_gray.shape[1]):
            pixel_value = frame_gray[i, j]
            ax.text(j, i, f"{pixel_value}", ha='center', va='center', color='black')

    plt.show()

if __name__ == '__main__':
    args = sys.argv # List of arguments, e.g: ['Ex2b-ChangeDetectionInVideosMyExercise.py', '0.95', '10', '15000', 'True']
    alpha, T, A, ex6 = np.array(args[1:]).astype(np.float32) # Convert to float32
    ex6 = bool(ex6) # Convert to bool the last argument
    capture_from_camera_and_show_images(alpha, T, A, ex6)
