import sys
import numpy as np

import pydicom
import pandas as pd
from src_data.mask_functions import mask2rle , rle2mask
import matplotlib.pyplot as plt
import cv2
from skimage import morphology, io , color , exposure , img_as_float , transform
from skimage.morphology.misc import remove_small_holes



def get_lung_seg_tensor(file_path, batch_size, seg_size, n_channels):
    X = np.empty((batch_size, seg_size, seg_size , n_channels))

    pixels_array = pydicom.read_file(file_path).pixel_array
    image_resized = cv2.resize(pixels_array, (seg_size, seg_size))
    image_resized = exposure.equalize_hist(image_resized)
    image_resized = np.array(image_resized, dtype= np.float64)
    image_resized -= image_resized.mean()
    image_resized /= image_resized.std()

    X[0,] = np.expand_dims(image_resized, axis = 2)

    return X



def remove_small_region(img , size):
    img = morphology.remove_small_objects(img, size)
    img = morphology.remove_small_holes(img, size)

    return img


def bounding_box(img):

    rows = np.any(img, axis= 1 )
    cols = np.any(img, axis= 0 )

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax , cmin , cmax


def get_lung_seg_rle(metadata_df, seg_size , lung_seg_model):
    processed_image = []
    for id , row in metadata_df.iterrows():

        img = get_lung_seg_tensor(row['file_path'], 1 , seg_size , 1 )

        seg_mask = lung_seg_model.predict(img).reshape((seg_size, seg_size))

        seg_mask = seg_mask > 0.5

        seg_mask = remove_small_region(seg_mask, 0.02 * np.prod(seg_size))

        processed_img = {}
        processed_img['id'] = row['id']
        processed_img['lung_mask'] = mask2rle(seg_mask * 255 , seg_size , seg_size)
        processed_img['rmin'] , processed_img['rmax'] , processed_img['cmin'] , processed_img['cmax'] = bounding_box(seg_mask)
        processed_image.append(processed_img)

    return pd.DataFrame(processed_image)



def plot_lung_seg(file_path , mask_encoded_list , lung_mask , rmin , rmax , cmin , cmax):
    pixel_array = pydicom.dcmread(file_path).pixel_array

    mask_dencoded_list = [rle2mask(mask_encoded, 1024 , 1024).T for mask_encoded in mask_encoded_list]
    lung_mask_decoded = cv2.resize(rle2mask(lung_mask, 256 , 256),(1024, 1024))

    rmin , rmax , cmin , cmax = rmin * 4, rmax * 4 , cmin * 4 , cmax * 4

    fig , ax = plt.subplots(nrows= 1 , ncols=3 , sharey=True , figsize = (20,10))

    ax[0].imshow(pixel_array, cmap = plt.cm.bone)
    ax[0].imshow(lung_mask_decoded, alpha = 0.3 , cmap = 'Blues' )
    ax[0].set_title('Check Xray with Lung Mask')

    ax[1].imshow(pixel_array[rmin:rmax + 1 , cmin : cmax + 1], cmap = plt.cm.bone)
    ax[1].set_title('Cropped Xray')
    
    ax[2].imshow(lung_mask_decoded, cmap = 'Blues')
    
    for mask_decoded in mask_dencoded_list:
        ax[2].imshow(mask_decoded, alpha = 0.3 , cmap = 'Reds')

    ax[2].set_title('Lung Mask With Pneumothorax')

    
