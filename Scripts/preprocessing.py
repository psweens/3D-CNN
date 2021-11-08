import os
import cv2
import glob
import numpy as np
import scipy as sp
import skimage as io
from view3D import ani3D
from skimage import io
from func_utils import natural_keys, norm_data, load_volume, resize_stacks, stdnorm
import matplotlib.pyplot as plt

def resize_as_npy(im_path, img_size=(600,600,700), target_size=(128,128,128),
              vessel=False, subvol=False, subvol_size=None, save_tiffs=False):
    
    stack_IDs = os.listdir(im_path)
    
    print('3D image stacks to be downsampled to ... (%d, %d, %d)' 
              %(target_size[0], target_size[1], target_size[2]))
    
    for i, ID in enumerate(stack_IDs):
        
        img_in_path = os.path.join(im_path, ID, "Input/")
        imgs_in = load_volume(img_in_path, size=img_size, datatype='uint8',)
        
        img_mask_path = os.path.join(im_path, ID, "roi_Mask/")
        imgs_mask = load_volume(img_mask_path, size=img_size, datatype='uint8')
        
        #  Weiner filter for input image
        # for j in range(0, target_size[2]):
        #     imgs_in[:,:,j] = sp.signal.wiener(imgs_in[:,:,j],10)
        
        
        if save_tiffs:
            inarrayout = os.path.join(im_path, ID, ID+'_readin_input.tiff')
            io.imsave(inarrayout, np.transpose(imgs_in.astype('uint8'),(2,0,1)), bigtiff=False)
            
            maskarrayout = os.path.join(im_path, ID, ID+'_readin_mask.tiff')
            io.imsave(maskarrayout, np.transpose(imgs_mask.astype('uint8'),(2,0,1)), bigtiff=False)
            

        #  Downsampling image volumes
        imgs_in, imgs_mask = resize_stacks(imgs_in.astype('uint8'), imgs_mask, img_size=img_size, 
                                          target_size=target_size)
        
        #  Apply local standardisation
        imgs_in = imgs_in.astype('float32')
        # for j in range(imgs_in.shape[2]):
        #     imgs_in[:,:,j] = stdnorm(imgs_in[:,:,j])
        
        lp = sp.stats.scoreatpercentile(imgs_in,5)
        imgs_in[imgs_in < lp] = lp
        up = sp.stats.scoreatpercentile(imgs_in,95)
        imgs_in[imgs_in > up] = up

        imgs_in = norm_data(imgs_in.astype('float32'))  
        imgs_mask = imgs_mask.astype('float32') / 255.0
        

        inarrayout = os.path.join(im_path, ID, ID+'_input.npy')
        np.save(inarrayout, imgs_in)
        maskarrayout = os.path.join(im_path, ID, ID+'_mask.npy')
        np.save(maskarrayout, imgs_mask)
        
        if save_tiffs:
            inarrayout = os.path.join(im_path, ID, ID+'_input.tiff')
            io.imsave(inarrayout, np.transpose((imgs_in*255).astype('uint8'),(2,0,1)), bigtiff=False)
            
            maskarrayout = os.path.join(im_path, ID, ID+'_mask.tiff')
            io.imsave(maskarrayout, np.transpose((imgs_mask*255).astype('uint8'),(2,0,1)), bigtiff=False)
            
        print('Saved %d of %d ... %s' %(i+1, len(stack_IDs), ID))
        

