import numpy as np
from skimage.morphology import erosion, dilation, binary_erosion, opening, closing, white_tophat, reconstruction, black_tophat, skeletonize, convex_hull_image, thin, remove_small_objects, remove_small_holes
from skimage.morphology import square, diamond, octagon, rectangle, star, disk
from skimage.filters.rank import entropy, enhance_contrast_percentile
from PIL import Image, ImageEnhance
from scipy import ndimage as ndi
from skimage.util import img_as_ubyte
from skimage.exposure import equalize_adapthist
import math
from skimage import data, filters, util
from scipy.signal import convolve2d
from matplotlib import pyplot as plt
import cv2

def my_segmentation(img, img_mask):
    # inversion des couleurs et on enleve l'anneau
    img_greyscale = (255*img_mask & (np.invert(img)))
    unique2, counts2 = np.unique(img_greyscale, return_counts=True)

    # VEINES -------------------------------------------------------------------
    img_tophat = white_tophat(img_greyscale, star(5))


    # plot des diagrammes
    unique, counts = np.unique(img_tophat, return_counts=True)
    #plt.bar(unique2, counts2)
    #plt.show()
    #plt.bar(unique, counts)

    # seuil de l'intensité
    seuil = 25
    img_seuil = img_tophat > seuil

    img_venules = img_seuil

    seuil_uint8 = (img_seuil * 255).astype(np.uint8)
        
    # Find connected components
    num_labels, labels = cv2.connectedComponents(seuil_uint8)
    
    img_venules = np.zeros_like(img_seuil)
    
    for label in range(1, num_labels):
        component = (labels == label)
        component_size = np.sum(component)
        
        # Keep only components of reasonable size (filter out noise)
        if component_size > 30:  # Minimum length for capillaries
            # Additional refinement can be added here
            img_venules |= component

    # CAPILLAIRES --------------------------------------------------------------
    # centerline candidates highlighting
    img_capil = np.zeros((512, 512))
    img_contrast = equalize_adapthist(img)

    for i in range(0, 181, 10):
        img_rota1 = ndi.rotate(img_contrast, i)
        img_rota= util.invert(img_rota1)
        img_convolve = filters.difference_of_gaussians(img_rota, low_sigma= 1)
        img_straight = ndi.rotate(img_convolve, -i)
        
        w, h = img_straight.shape

        left = int(np.ceil((w - 512) / 2))
        right = w - int(np.floor((w - 512) / 2))

        top = int(np.ceil((h - 512) / 2))
        bottom = h - int(np.floor((h - 512) / 2))

        img_cropped = img_straight[top:bottom, left:right]
        img_capil += img_cropped
    
    img_capillaires = img_capil > 0.2
    skeleton = skeletonize(img_capillaires)

    centerlines_uint8 = (skeleton * 255).astype(np.uint8)
        
    # Find connected components
    num_labels, labels = cv2.connectedComponents(centerlines_uint8)
    
    refined_centerlines = np.zeros_like(skeleton)
    
    for label in range(1, num_labels):
        component = (labels == label)
        component_size = np.sum(component)
        
        # Keep only components of reasonable size (filter out noise)
        if component_size > 30:  # Minimum length for capillaries
            # Additional refinement can be added here
            refined_centerlines |= component
    
    

    img_preout = 255*img_venules + 255*refined_centerlines
    img_preout = np.clip(img_preout, 0, 1)

    circle_mask = np.zeros_like(img_preout, np.uint8)
    circle_mask = cv2.circle(circle_mask, (256, 256), 253, (1, 1, 1), -1)

    img_out = img_preout & circle_mask


    plt.subplot(131)
    plt.imshow(img_preout, cmap='gray')
    plt.subplot(132)
    plt.imshow(circle_mask, cmap='gray')
    plt.subplot(133)
    plt.imshow(img_out, cmap='gray')
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
img =  np.asarray(Image.open('./images_IOSTAR/star02_OSC.jpg')).astype(np.uint8)
print(img.shape)

nrows, ncols = img.shape
row, col = np.ogrid[:nrows, :ncols]
#On ne considere que les pixels dans le disque inscrit 
img_mask = (np.ones(img.shape)).astype(np.bool_)
invalid_pixels = ((row - nrows/2)**2 + (col - ncols/2)**2 > (nrows / 2)**2)
img_mask[invalid_pixels] = 0

img_out = my_segmentation(img,img_mask)

#Ouvrir l'image Verite Terrain en booleen
img_GT =  np.asarray(Image.open('./images_IOSTAR/GT_02.png')).astype(np.uint8)

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

