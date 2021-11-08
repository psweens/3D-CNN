import os
import re
import pickle
import glob
import shutil
import cv2
import csv
import numpy as np
import itk
import keras.initializers as ki

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def deletecontents(mydir):
    filelist = glob.glob(os.path.join(mydir, "*.tiff"))
    for f in filelist:
       os.remove(f)
        
def norm_data(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))

def stdnorm(data):
    return (data - np.mean(data)) / np.std(data)

def save_dict(di_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(di_, f)

def load_dict(filename_):
    with open(filename_, 'rb') as f:
        ret_di = pickle.load(f)
    return ret_di

def load_volume(folder, size=(600,600,700), ext='*.tiff', datatype='float32',
                stack=False):
    
    vol = np.empty([size[0], size[1], size[2]], dtype=datatype)
    
    if ext == '*.npy':
        imgs = os.path.join(folder, ext)
        vol = np.load(imgs)
    else:
        if stack:
            # imgs = os.path.join(folder, 'mask'+ext)
            imgs = folder + ext
            vol = (np.array(itk.imread(imgs, itk.F))).astype(datatype)
        else:
            imgs = glob.glob(os.path.join(folder, ext))
            imgs.sort(key=natural_keys)
            for i in range(0,size[2]):
                idx = imgs[i]
                img = cv2.imread(idx, cv2.IMREAD_GRAYSCALE)
                vol[:,:,i] = img
        
    return vol

def resize_stacks(input_imgs, mask_imgs, img_size=None, target_size=None):
    
    A = np.empty([target_size[0], target_size[1], img_size[2]],
                           dtype='uint8')
    B = np.empty([target_size[0], target_size[1], img_size[2]],
                           dtype='uint8')
    
    X = np.empty([target_size[0], target_size[1], target_size[2]],
                           dtype='uint8')
    Y = np.empty([target_size[0], target_size[1], target_size[2]],
                 dtype='uint8')
    
    #  Assuming input and mask images are equal in size
    if target_size[0] > img_size[0] & target_size[1] > img_size[1]:
        imethod = cv2.INTER_AREA
    else:
        imethod = cv2.INTER_CUBIC

    for j in range(0, img_size[2]):
    
        A[:,:,j] = cv2.resize(input_imgs[:,:,j], (target_size[0], target_size[1]),
                              interpolation = imethod)
        
        B[:,:,j] = cv2.resize(mask_imgs[:,:,j], (target_size[0], target_size[1]),
                              interpolation = imethod)
    
    
    if target_size[1] > img_size[1] & target_size[2] > img_size[2]:
        imethod = cv2.INTER_AREA
    else:
        imethod = cv2.INTER_CUBIC

    for j in range(0, target_size[0]):
        
        X[j,] = cv2.resize(A[j,], (target_size[2], target_size[1]),
                           interpolation = imethod)
                                            
        Y[j,] = cv2.resize(B[j,], (target_size[2], target_size[1]),
                           interpolation = imethod)
        
        cv2.threshold(Y[j,], 10, 255, cv2.THRESH_BINARY)
            
    return X, Y

def resize_volume(img, target_size=None):
    
    arr1 = np.empty([target_size[0], target_size[1], img.shape[2]], dtype='uint8')
    arr2 = np.empty([target_size[0], target_size[1], target_size[2]], dtype='uint8')
    
    for i in range(img.shape[2]):
        arr1[:,:,i] = cv2.resize(img[:,:,i], (target_size[0], target_size[1]),
                                 interpolation=cv2.INTER_CUBIC)
        
    for i in range(target_size[0]):
        arr2[i,:,:] = cv2.resize(arr1[i,], (target_size[2], target_size[1]),
                                 interpolation=cv2.INTER_CUBIC)
    
    for i in range(arr2.shape[2]):
        _, arr2[:,:,i] = cv2.threshold(arr2[:,:,i], 127, 255, cv2.THRESH_BINARY)
        
    return arr2

def dict_csv(csv_file, mydict):
    
    with open(csv_file, 'w', newline="", encoding='utf-8-sig') as csv_file:  
        writer = csv.writer(csv_file)
        for key, value in mydict.items():
            writer.writerow([key])
            writer.writerows(map(lambda x: [x], value))

def initializer_out(name, seed=None):
    
    out = []
    if name == 'lecun_normal':
        out = ki.lecun_normal(seed=None)
    elif name == 'he_uniform':
        out = ki.he_uniform(seed=None)
        
    return out

