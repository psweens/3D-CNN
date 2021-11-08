import os
import cv2
import glob
import math as m
import numpy as np
import matplotlib.pyplot as plt
import random
from keras.preprocessing.image import save_img
from customDataGen import DataGenerator
from func_utils import deletecontents, norm_data, natural_keys, load_volume, resize_volume
from view3D import ani3D
from skimage import io

def testCNN(model, partition, img_size, target_size, im_path, label_path, 
            pred_dir, batch_size=1, n_classes = 2, subvolumes=False):
    
    params = {'img_dim': img_size,
              'dim': target_size,
              'batch_size': batch_size,
              'im_path': im_path,
              'label_path': label_path,
              'n_classes': n_classes,
              'shuffle': False,
              'augment': False,
              'subvolumes': subvolumes,
              'subvol_dim': target_size,
              'testing': True}
    
    deletecontents('Output/Predictions/')

    test_generator = DataGenerator(partition['testing'], **params)
    prediction = model.predict_generator(test_generator, 
                                         steps=len(partition['testing']), 
                                         verbose = 1)
    
    prediction = (255 * prediction).astype('uint8')
    batch_dir = os.path.join(pred_dir, 'Output/Predictions/')
    for i in range(prediction.shape[0]):
        
        # Upscale image
        arr = resize_volume(prediction[i,0,].astype('uint8'), img_size)
        io.imsave(batch_dir+"{file}_upscaled.tiff".format(file=partition['testing'][i]), 
                  np.transpose(arr,(2,0,1)), 
                  bigtiff=False)
        np.save(batch_dir+"{file}.npy".format(file=partition['testing'][i]), prediction[i,0,:,:,:])

        # Save CNN prediction
        io.imsave(batch_dir+"{file}.tiff".format(file=partition['testing'][i]), 
                  np.transpose(prediction[i,0,:,:,:],(2,0,1)), 
                  bigtiff=False)


    return prediction

def saveimagevol(batch_dir,img_pred, thresh=False, cutoff = 150):
    
    for j in range(img_pred.shape[3]):
        
            saveimg = img_pred[...,j]
            if thresh:
                ret, saveimg[0,] = cv2.threshold(saveimg[0,], cutoff, 255, cv2.THRESH_BINARY)
            
            dir_string = batch_dir + '/' + str(j) + '.tiff'
            save_img(dir_string, saveimg, data_format = 'channels_first', scale = [0, 255])

    return img_pred

def upscaleimg(imgvol, targetsize, savedir):
    
    # Channel first
    X = np.empty([1,targetsize[0], targetsize[1], targetsize[2]])
    Y = np.empty([1,targetsize[0], targetsize[1], imgvol.shape[3]])
    
    # Below uses GANs to increase image size
    # model = RRDN(weights='gans')
    
    imgvol[0,] = imgvol[0,] * 255
    
    for i in range(0, imgvol.shape[3]):
        # Y[:,:,i] = cv2.resize(imgvol[:,:,i], (targetsize[0], targetsize[1]),
                              # interpolation = cv2.INTER_CUBIC)
        # img = cv2.cvtColor(imgvol[:,:,i].astype('float32'),cv2.COLOR_GRAY2RGB)
        # img = model.predict(img)
        # img = model.predict(img)
        # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        # img = cv2.equalizeHist(img[:,:,0])
        # img = cv2.GaussianBlur(img,(5,5),0)
        # cv2.threshold(img, 150, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        
        
        Y[0,:,:,i] = cv2.resize(imgvol[0,:,:,i], (targetsize[0], targetsize[1]),
                              interpolation = cv2.INTER_CUBIC)
        # cv2.threshold(Y[0,:,:,i], 230, 255, cv2.THRESH_BINARY)
        
    for i in range(0, targetsize[1]):
        # X[0,i,] = cv2.resize(Y[i,:,:], (targetsize[2], targetsize[1]),
                           # interpolation = cv2.INTER_CUBIC)
        # img = cv2.cvtColor(Y[i,].astype('float32'),cv2.COLOR_GRAY2RGB)
        # img = model.predict(img)
        # img = model.predict(img)
        # img = cv2.cvtColor(channelavg(img), cv2.COLOR_RGB2GRAY)
        X[0,i,] = cv2.resize(Y[0,i,], (targetsize[2], targetsize[1]),
                         interpolation = cv2.INTER_CUBIC)
        # cv2.threshold(X[0,i,], 230, 255, cv2.THRESH_BINARY)
    
    # norm_data(X, 255)
    X = saveimagevol(savedir, X, thresh=True, cutoff=100)

    return X

def drawROIs(X, targetsize, contourdir, exp_dir, binary = False):
    
    if binary:
       imgs_in = glob.glob(os.path.join(exp_dir, "*.tif"))
       bin_img = io.imread(imgs_in[0])
       bin_img = bin_img.transpose(1,2,0)
       
    else:
            
        input_dir = os.path.join(exp_dir,'Input/')
        imgs_in = glob.glob(os.path.join(input_dir, "*.tiff"))
        imgs_in.sort(key=natural_keys)
                             
    label_dir = os.path.join(exp_dir,'Mask/')
    imgs_mask = glob.glob(os.path.join(label_dir, "*.tiff"))
    imgs_mask.sort(key=natural_keys)
    
    Y = np.empty([targetsize[0], targetsize[1], 1], dtype='uint8')
    os.mkdir(contourdir)
    for i in range(0, X.shape[3]):
        
        if binary:
            img = cv2.cvtColor(bin_img[:,:,i], cv2.COLOR_GRAY2RGB)
        else:
            idx = imgs_in[i]
            img = cv2.imread(idx, cv2.IMREAD_UNCHANGED)
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        
        idx = imgs_mask[i]
        mask = cv2.imread(idx, cv2.IMREAD_UNCHANGED)
        (cmask, _) = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        Y[:,:,0] = X[0,:,:,i].astype('uint8')
        (contours, _) = cv2.findContours(Y, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        mask = np.ones(Y.shape[:2], dtype="uint8") * 255
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        
        cv2.drawContours(img, contours, -1, color=(255,0,0), thickness=5)
        cv2.drawContours(img, cmask, -1, color=(0,255,0), thickness=5)
        cv2.imwrite(contourdir + str(i) + '.tiff', img)

def channelavg(img):
    
    avg = 0
    X = np.empty([img.shape[0], img.shape[1]])
    for k in range(0,img.shape[1]):
        for j in range(0, img.shape[0]):
            avg = 0
            for i in range(0,3):
                avg = avg + img[j,k,i]
            X[j,k] = avg / 3
            
    return X