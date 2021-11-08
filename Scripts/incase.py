import os
import cv2
import glob
import numpy as np
from view3D import ani3D
from skimage import io
from func_utils import natural_keys, norm_data, load_volume
import matplotlib.pyplot as plt

def saveasnpy(im_path, img_size=(600,600,700), target_size=(128,128,128),
              vessel=False, subvol=False):
    
    stack_IDs = os.listdir(im_path)
    
    for i, ID in enumerate(stack_IDs):
        img_in_path = os.path.join(im_path, ID, "Input/")
        imgs_in = glob.glob(os.path.join(img_in_path, "*.tiff"))
        imgs_in.sort(key=natural_keys)
    
        img_mask_path = os.path.join(im_path, ID, "roi_Mask/")
        imgs_mask = glob.glob(os.path.join(img_mask_path, "*.tiff"))
        imgs_mask.sort(key=natural_keys)

        if vessel:
            bin_in = glob.glob(os.path.join(im_path, ID, "vess_Mask/*.tif"))
            bin_input = io.imread(bin_in[0])
            bin_input = bin_input.transpose(1,2,0)
            bin_inVol = np.empty([target_size[0], target_size[1], img_size[2]],
                             dtype='uint8')
            Z = np.empty([target_size[0], target_size[1], target_size[2]])
            
        
        img_inVol = np.empty([target_size[0], target_size[1], img_size[2]],
                             dtype='uint8')
        img_maskVol = np.empty([target_size[0], target_size[1], img_size[2]],
                               dtype='uint8')
        
        X = np.empty([target_size[0], target_size[1], target_size[2]])
        Y = np.empty([target_size[0], target_size[1], target_size[2]])
        for j in range(0, img_size[2]):
            idx = imgs_in[j]
            img_inVol[:,:,j] = cv2.resize(cv2.imread(idx, cv2.IMREAD_UNCHANGED), 
                                      (target_size[0], target_size[1]),
                                      interpolation = cv2.INTER_AREA)
            
            jdx = imgs_mask[j]
            img_maskVol[:,:,j] = cv2.resize(cv2.imread(jdx, cv2.IMREAD_UNCHANGED), 
                                      (target_size[0], target_size[1]),
                                      interpolation = cv2.INTER_AREA)
            
            if vessel:
                bin_inVol[:,:,j] = cv2.resize(bin_input[:,:,j], 
                                      (target_size[0], target_size[1]),
                                      interpolation = cv2.INTER_AREA)
                
            
        for j in range(0, target_size[0]):
            X[j,] = cv2.resize(img_inVol[j,], (target_size[2], target_size[1]),
                                                interpolation = cv2.INTER_AREA)
                                                
            Y[j,] = cv2.resize(img_maskVol[j,], (target_size[2], target_size[1]),
                                                interpolation = cv2.INTER_AREA)
            cv2.threshold(Y[j,], 10, 255, cv2.THRESH_BINARY)
            
            if vessel:
                Z[j,] = cv2.resize(bin_inVol[j,], (target_size[2], target_size[1]),
                                                interpolation = cv2.INTER_AREA)
                cv2.threshold(Z[j,], 10, 255, cv2.THRESH_BINARY)
        
        X = norm_data(data=X)
        Y = norm_data(data=Y)
        Y.astype('uint8')

        if vessel:
            Z = norm_data(data=Z)
            binarrayout = os.path.join(im_path, ID, ID+'_mask.npy')
            np.save(binarrayout, Z)
        else:
            maskarrayout = os.path.join(im_path, ID, ID+'_mask.npy')
            np.save(maskarrayout, Y)
            
        if subvol:
            get_subVolumes(X,Z,im_path,ID)
            
       
        inarrayout = os.path.join(im_path, ID, ID+'_input.npy')
        
        
        np.save(inarrayout, X)
        
        
        print('Saved %d of %d ...' %(i+1, len(stack_IDs)))
        
        
        
def get_subVolumes(input_stack, mask_stack, image_path, ID,
                   subvol_size=(256,256,128)):        
    
    im_path = os.path.join(image_path, ID, 'Subvolumes')
    
    overlap_size = (input_stack.shape[0] % subvol_size[0],
                    input_stack.shape[1] % subvol_size[1],
                    input_stack.shape[2] % subvol_size[2])
    # print('%f %f %f' %(overlap_size[0], overlap_size[1], overlap_size[2]))
    
    if not os.path.exists(im_path):
        os.mkdir(im_path)
        os.mkdir(os.path.join(im_path, 'Input'))
        os.mkdir(os.path.join(im_path, 'Mask'))
    
    # print(np.arange(0, input_stack.shape[0]-subvol_size[0],
    #                    subvol_size[0]-overlap_size[0]))
    # print(np.arange(0, input_stack.shape[2]-subvol_size[2],
    #                    subvol_size[2]-overlap_size[2]))
    for x in np.arange(0, input_stack.shape[0]-subvol_size[0],
                       subvol_size[0]-overlap_size[0]):
        for y in np.arange(0, input_stack.shape[1]-subvol_size[1],
                       subvol_size[1]-overlap_size[1]):
            for z in np.arange(0, input_stack.shape[2]-subvol_size[2],
                       subvol_size[2]-overlap_size[2]):

                input_subvol = input_stack[x:x+subvol_size[0],
                                            y:y+subvol_size[1],
                                            z:z+subvol_size[2]]
                
                # plt.figure()
                # plt.imshow(input_subvol[...,64])
                
                mask_subvol = mask_stack[x:x+subvol_size[0],
                                          y:y+subvol_size[1],
                                          z:z+subvol_size[2]]
                
                # plt.figure()
                # plt.imshow(mask_subvol[...,64])
                
                filename = str(x+subvol_size[0]) + '_' + str(y+subvol_size[1]) + '_' + str(z+subvol_size[2]) + '_' + ID
                
                
                inputarrayout = os.path.join(im_path, 'Input/', 
                                            filename+'.npy')
                np.save(inputarrayout, input_subvol)
                
                maskarrayout = os.path.join(im_path, 'Mask/', 
                                            filename+'.npy')
                np.save(maskarrayout, mask_subvol)
                