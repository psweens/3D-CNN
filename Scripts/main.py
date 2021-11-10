import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import numpy as np
import keras_metrics as km
import keras.initializers as ki
import tensorflow as tf

from keras.callbacks import ModelCheckpoint, CSVLogger
from keras.models import load_model
from keras.utils import plot_model
from keras.callbacks import EarlyStopping
from unet_model import Unet3D
from vnet_model import Vnet
from test_3Dunet import testCNN
from customDataGen import run3DataGen
from configGPU import gpu_mem, _get_available_gpus
from custom_loss import dice_coef_loss, dice_coef, jaccard_distance
from hyper_optimisation import sweep_hyperparams
from plot_data import plot_metrics
from preprocessing import resize_as_npy
from func_utils import load_dict, save_dict
from analysis import analyse_predictions

# Check available GPU devices.
gpu_mem()
# _get_available_gpus()


# Input image and mask directories
image_path = ''
mask_path = ''

preprocess = False # True -> stack to npy arrays and downsample
subvolumes = False # Sample subvolumes?
loadcnn = True # True -> load previous model
hypersweep = False #  True -> optimise hyperparameters using Talos
load_data_partition = False # True -> Load previously partitioned dataset


# Parameters
batchsize = 1
nb_epochs = 120
img_size = (600, 600, 700)
target_size = (256,256,256)
subvolume_size = (256,256,256)
n_classes = 2


# Preprocess and resize data if necessary
if preprocess:
    resize_as_npy(image_path, img_size=img_size, target_size=target_size,
              subvol=subvolumes, subvol_size=subvolume_size, save_tiffs=False)


if loadcnn:    
    # Loadprevious model and data partition
    if load_data_partition:
        partition = load_dict('cnn_data_partition.pkl')
        
    model = load_model('3Dunet_rsom.hdf5',
                       custom_objects={'dice_coef_loss': dice_coef_loss,
                                       'dice_coef': dice_coef,
                                       'jaccard_distance': jaccard_distance,
                                       'binary_precision': km.binary_precision(),
                                       'binary_recall': km.binary_recall()})
    
    #  Print summary of model architecture
    model.summary()
    
else:
    
    if bool(subvolumes) == True:
        target_size = subvolume_size
    
    # Create generators and partition data
    # Partition details the image volume names for training/validation/testing data
    train_generator, validation_generator, partition = run3DataGen(image_path, mask_path, 
                                                                   batchsize,
                                                                   img_size=img_size,
                                                                   target_size=target_size,
                                                                   n_classes=n_classes,
                                                                   augment=False,
                                                                   load_pt=load_data_partition,
                                                                   subvolumes=subvolumes,
                                                                   subvol_dim=subvolume_size)
    
    #  Save training/validation/testing data partition
    save_dict(partition, 'Output/cnn_data_partition.pkl') 


    if hypersweep:
        #  Optimise hyperparameters
        sweep_hyperparams(train_generator, validation_generator,
                          target_size=target_size, batchsize=batchsize)
        
    else:
        
        # defines model_checkpoint which save the CNN model after each epoch
        csv_logger = CSVLogger('Output/training_log.csv', append=False, separator=';')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir="./logs")
        model_checkpoint = [ModelCheckpoint('Output/3Dunet_rsom.hdf5', monitor='val_dice_coef',
                                           verbose=1, save_best_only=False),
                            tensorboard_callback,
                            EarlyStopping(monitor='val_loss', patience=200,
                                          mode='min', min_delta=1e-4), csv_logger]
        
        #  Train model
        p = {'ds': 2,
             'dropout': 0.2,
             'epochs': nb_epochs,
             'lr': 1e-5,
             'loss': dice_coef_loss,
             'activation': 'relu',
             'decay': 1e-8,
             'k_initializer': 'glorot_uniform',
             'flayer': 'sigmoid'}
        history, model = Unet3D(train_generator, validation_generator,
                                target_size=target_size, batch_size=batchsize, 
                                callbacks=model_checkpoint, 
                                training_size=len(partition['train']), 
                                params=p)
        
        #  CNN training data
        np.save('Output/training_history.npy', history.history) 
        
        #  Plot metric data
        plot_metrics(history)


# model.summary()
plot_model(model, to_file='Output/3Dunet.png', show_shapes=True)


# Apply CNN to test data
# prediction = testCNN(model, partition, img_size=img_size, target_size=target_size,
#                       im_path=image_path, label_path=mask_path,
#                       pred_dir=os.getcwd(),
#                       n_classes=n_classes, subvolumes=subvolumes)

#  Compare predictions against other
vess_path = ''
files = os.listdir(mask_path)
analyse_predictions(files,
                    os.getcwd(),
                    mask_path,
                    vess_path,
                    name='Manual')





