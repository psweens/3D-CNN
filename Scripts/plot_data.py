import os
import skimage
import glob
import re
import matplotlib.pyplot as plt
import numpy as np

from skimage import io
from func_utils import natural_keys

def plot_input(im_path, im_ID, subvolumes=False, img_size=None):
    
    if subvolumes:
        val = img_size[0] / 2
        val = str(344)
        imgs = os.path.join(im_path, im_ID[0], 'Subvolumes/Input',
                            val+'_'+val+'_'+val+'_'+im_ID[0]+'.npy')
        imgs = np.load(imgs)
        masks = os.path.join(im_path, im_ID[0], 'Subvolumes/Mask',
                            val+'_'+val+'_'+val+'_'+im_ID[0]+'.npy')
        masks = np.load(masks)
    else:
        imgs = os.path.join(im_path, im_ID[0], im_ID[0]+"_input.npy")
        imgs = np.load(imgs)
        masks = os.path.join(im_path, im_ID[0], im_ID[0]+"_mask.npy")
        masks = np.load(masks)
    
    plt.figure(1)
    for i in range(1, 10):
        plt.subplot(3, 3, i)
        plt.imshow(imgs[:,:,64], cmap=plt.get_cmap('gray'))
    plt.suptitle(im_ID[0]) 
    
    plt.figure(2)
    for i in range(1, 10):
        plt.subplot(3, 3, i)
        plt.imshow(masks[:,:,64], cmap=plt.get_cmap('gray'))
    plt.suptitle(im_ID[0])    

def plot_tiff(im_path, im_ID):   
    
    path = os.path.join(im_path, im_ID[0])
    
    input_path = os.path.join(path, "Input/")
    input_images = glob.glob(os.path.join(input_path, "*.tiff"))
    input_images.sort(key=natural_keys)
    
    mask_path = os.path.join(path, "Mask/")
    mask_images = glob.glob(os.path.join(mask_path, "*.tiff"))
    mask_images.sort(key=natural_keys)
    
    plt.figure(1)
    for i in range(1, 10):
        img_X = skimage.io.imread(input_images[300+i]) / 255
        plt.subplot(3, 3, i)
        plt.imshow(img_X, cmap=plt.get_cmap('gray'))
    plt.suptitle(im_ID[0]) 
    
    plt.figure(2)
    for i in range(1, 10):
        img_Y = skimage.io.imread(mask_images[300+i]) / 255
        plt.subplot(3, 3, i)
        plt.imshow(img_Y, cmap=plt.get_cmap('gray'))
    plt.suptitle(im_ID[0])    
        
    
def plot_metrics(history):
    
    # Plot training & validation accuracy values
    plt.figure(3)
    plt.plot(history.history['dice_coef'])
    plt.plot(history.history['val_dice_coef'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    leg = plt.legend(['Train', 'Validation'], loc='lower right')
    leg.get_frame().set_linewidth(0.0)
    plt.savefig('Output/Model_Accuracy.pdf')
    plt.show()
    
    
    # Plot training & validation loss values
    plt.figure(4)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    leg = plt.legend(['Train', 'Validation'], loc='upper right')
    leg.get_frame().set_linewidth(0.0)
    plt.savefig('Output/Model_Loss.pdf')
    plt.show()
