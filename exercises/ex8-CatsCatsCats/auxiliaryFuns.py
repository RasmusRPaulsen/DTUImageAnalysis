import numpy as np
import matplotlib.pyplot as plt
from skimage import io
from skimage import img_as_ubyte
from skimage.transform import SimilarityTransform, warp
import glob
import os

def read_landmark_file(file_name):
    f = open(file_name, 'r')
    lm_s = f.readline().strip().split(' ')
    n_lms = int(lm_s[0])
    if n_lms < 3:
        print(f"Not enough landmarks found")
        return None

    new_lms = 3
    # 3 landmarks each with (x,y)
    lm = np.zeros((new_lms, 2))
    for i in range(new_lms):
        lm[i, 0] = lm_s[1 + i * 2]
        lm[i, 1] = lm_s[2 + i * 2]
    return lm

def create_u_byte_image_from_vector(im_vec, height, width, channels):
    min_val = im_vec.min()
    max_val = im_vec.max()

    # Transform to [0, 1]
    im_vec = np.subtract(im_vec, min_val)
    im_vec = np.divide(im_vec, max_val - min_val)
    im_vec = im_vec.reshape(height, width, channels)
    im_out = img_as_ubyte(im_vec)
    return im_out

def align_and_crop_one_cat_to_destination_cat(img_src, lm_src, img_dst, lm_dst):
    """
    Landmark based alignment of one cat image to a destination
    :param img_src: Image of source cat
    :param lm_src: Landmarks for source cat
    :param lm_dst: Landmarks for destination cat
    :return: Warped and cropped source image. None if something did not work
    """
    tform = SimilarityTransform()
    tform.estimate(lm_src, lm_dst)
    warped = warp(img_src, tform.inverse, output_shape=img_dst.shape)

    # Center of crop region
    cy = 185
    cx = 210
    # half the size of the crop box
    sz = 180
    warp_crop = warped[cy - sz:cy + sz, cx - sz:cx + sz]
    shape = warp_crop.shape
    if shape[0] == sz * 2 and shape[1] == sz * 2:
        return img_as_ubyte(warp_crop)
    else:
        print(f"Could not crop image. It has shape {shape}. Probably to close to border of image")
        return None

def preprocess_one_cat():
    src = "data/MissingCat"
    dst = "data/ModelCat"
    out = "data/MissingCatProcessed.jpg"

    src_lm = read_landmark_file(f"{src}.jpg.cat")
    dst_lm = read_landmark_file(f"{dst}.jpg.cat")

    src_img = io.imread(f"{src}.jpg")
    dst_img = io.imread(f"{dst}.jpg")

    src_proc = align_and_crop_one_cat_to_destination_cat(src_img, src_lm, dst_img, dst_lm)
    if src_proc is None:
        return

    io.imsave(out, src_proc)

    fig, ax = plt.subplots(ncols=3, figsize=(16, 6))
    ax[0].imshow(src_img)
    ax[0].plot(src_lm[:, 0], src_lm[:, 1], '.r', markersize=12)
    ax[1].imshow(dst_img)
    ax[1].plot(dst_lm[:, 0], dst_lm[:, 1], '.r', markersize=12)
    ax[2].imshow(src_proc)
    for a in ax:
        a.axis('off')
    plt.tight_layout()
    plt.show()
    

import os
import glob
from skimage import io
from skimage import img_as_ubyte

# Suponiendo que tienes estas funciones definidas correctamente en otro lugar:
# read_landmark_file, align_and_crop_one_cat_to_destination_cat

def preprocess_all_cats(in_dir, out_dir):
    """
    Create aligned and cropped version of image
    :param in_dir: Where are the original photos and landmark files
    :param out_dir: Where should the preprocessed files be placed
    """
    # Ruta de la imagen de referencia (modelo)
    dst = "D:\\First semester\\Imagenes\\PRACTICAS\\ImageAnalysis\\exercises\\ex8-CatsCatsCats\\data\\ModelCat"  # Sin extensión, ya que leeremos ModelCat.jpg directamente.

    # Lee los puntos de referencia (landmarks) de la imagen de destino
    dst_lm = read_landmark_file(f"{dst}.jpg.cat")
    #print(dst_lm)
    # Carga la imagen de referencia (modelo)
    dst_img = io.imread(f"{dst}.jpg")
    
    print("Imagen de destino cargada correctamente.")
    
    # Obtiene todas las imágenes que realmente se buscan procesar
    all_images = glob.glob(os.path.join(in_dir, "*.jpg"))
    
    if not all_images:
        print("No se encontraron imágenes en el directorio.")
        return

    # Iterar sobre todas las imágenes y procesarlas
    for img_idx in all_images:
        # El nombre sin la extensión
        name_no_ext = os.path.splitext(img_idx)[0]
        base_name = os.path.basename(name_no_ext)
        
        # Define el nombre de salida para la imagen procesada
        out_name = os.path.join(out_dir, f"{base_name}_preprocessed.jpg")
        
        # Lee los puntos de referencia (landmarks) de la imagen de entrada
        src_lm = read_landmark_file(f"{name_no_ext}.jpg.cat")
        
        # Carga la imagen original
        src_img = io.imread(f"{name_no_ext}.jpg")
        
        # Alinea y recorta la imagen
        proc_img = align_and_crop_one_cat_to_destination_cat(src_img, src_lm, dst_img, dst_lm)
        
        # Si la imagen procesada no es None, guárdala
        if proc_img is not None:
            io.imsave(out_name, img_as_ubyte(proc_img))  # Asegúrate de convertir la imagen a uint8 antes de guardarla
            print(f"Imagen procesada guardada: {out_name}")
        else:
            print(f"Error al procesar la imagen: {img_idx}")

