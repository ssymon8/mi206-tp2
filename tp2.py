import numpy as np
from skimage.morphology import erosion, dilation, binary_erosion, opening, closing, white_tophat, reconstruction, black_tophat, skeletonize, convex_hull_image, thin
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from skimage.filters.rank import entropy, enhance_contrast_percentile
from PIL import Image, ImageEnhance
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
from skimage.exposure import equalize_adapthist
import math
from skimage import data, filters
from scipy.signal import convolve2d
from matplotlib import pyplot as plt

def my_segmentation(img, img_mask, seuil):
    # inversion des couleurs et on enleve l'anneau
    img_greyscale = (255*img_mask & (np.invert(img)))
    unique2, counts2 = np.unique(img_greyscale, return_counts=True)

    # VEINES -------------------------------------------------------------------
    
    # white top hat
    #footprint = np.array([
    #    [0, 0, 1, 0, 0],
    #    [0, 0, 1, 0, 0],
    #    [1, 1, 1, 1, 1],
    #    [0, 0, 1, 0, 0],
    #    [0, 0, 1, 0, 0]
    #])
    footprint = np.array([
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 1, 1, 1, 0, 0]
    ])
    img_tophat = white_tophat(img_greyscale, star(5))


    # plot des diagrammes
    unique, counts = np.unique(img_tophat, return_counts=True)
    #plt.bar(unique2, counts2)
    #plt.show()
    #plt.bar(unique, counts)

    # seuil de l'intensité
    seuil = 25
    img_venules = img_tophat > seuil

    # CAPILLAIRES --------------------------------------------------------------
    # centerline candidates highlighting
    kernel = np.array([
        [-1, -2, 0, 2, 1],
        [-2, -4, 0, 4, 2],
        [-1, -2, 0, 2, 1]
    ])
    img_capil = np.zeros((512, 512))
    
    img_contrast = equalize_adapthist(img)

    for i in range(0, 181, 20):
        img_rota = ndi.rotate(img_contrast, i)
        img_convolve = convolve2d(img_rota, kernel)
        img_straight = ndi.rotate(img_convolve, -i)
        
        w, h = img_straight.shape

        left = int(np.ceil((w - 512) / 2))
        right = w - int(np.floor((w - 512) / 2))

        top = int(np.ceil((h - 512) / 2))
        bottom = h - int(np.floor((h - 512) / 2))

        img_cropped = img_straight[top:bottom, left:right]
        img_capil += img_cropped

    
    kernel_v = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ])

    #img_test = convolve2d(img_straight, kernel_v) > 25
    img_test = filters.sobel(img_capil)

    plt.subplot(121)
    plt.imshow(img, cmap='gray')
    plt.subplot(122)
    plt.imshow(img_capil, cmap='gray')
    plt.show()
    return img_out

def evaluate(img_out, img_GT):
    GT_skel = thin(img_GT,max_num_iter=15) # On reduit le support de l'evaluation...
    img_out_skel = thin(img_out, max_num_iter=15) # ...aux pixels des squelettes
    TP = np.sum(img_out_skel & img_GT) # Vrais positifs
    FP = np.sum(img_out_skel & ~img_GT) # Faux positifs
    FN = np.sum(GT_skel & ~img_out) # Faux negatifs

    ACCU = TP / (TP + FP) # Precision
    RECALL = TP / (TP + FN) # Rappel
    return ACCU, RECALL, img_out_skel, GT_skel

#Ouvrir l'image originale en niveau de gris
img =  np.asarray(Image.open('./images_IOSTAR/star01_OSC.jpg')).astype(np.uint8)
print(img.shape)

nrows, ncols = img.shape
row, col = np.ogrid[:nrows, :ncols]
#On ne considere que les pixels dans le disque inscrit 
img_mask = (np.ones(img.shape)).astype(np.bool_)
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2)
img_mask[invalid_pixels] = 0

img_out = my_segmentation(img,img_mask,80)

#Ouvrir l'image Verite Terrain en booleen
img_GT =  np.asarray(Image.open('./images_IOSTAR/GT_01.png')).astype(np.uint8)

ACCU, RECALL, img_out_skel, GT_skel = evaluate(img_out, img_GT)
print('Accuracy =', ACCU,', Recall =', RECALL)

plt.subplot(231)
plt.imshow(img,cmap = 'gray')
plt.title('Image Originale')
plt.subplot(232)
plt.imshow(img_out,cmap='gray')
plt.title('Segmentation')
plt.subplot(233)
plt.imshow(img_out_skel,cmap='gray')
plt.title('Segmentation squelette')
plt.subplot(235)
plt.imshow(img_GT,cmap='gray')
plt.title('Verite Terrain')
plt.subplot(236)
plt.imshow(GT_skel,cmap='gray')
plt.title('Verite Terrain Squelette')
plt.show()

