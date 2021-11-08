import os
import random
import keras
import numpy as np

from plot_data import plot_input
from aug_utils import random_augmentation
from func_utils import load_dict

def run3DataGen(im_path, label_path, batchsize, img_size=(600, 600, 700),
                target_size=(256, 256, 256), n_classes=2, load_pt=False,
                augment=False, subvolumes=False, subvol_dim=None):
    
    params = {'img_dim': img_size,
              'dim': target_size,
              'batch_size': batchsize,
              'im_path': im_path,
              'label_path': label_path,
              'n_classes': n_classes,
              'shuffle': True,
              'augment': augment,
              'subvolumes': subvolumes,
              'subvol_dim': subvol_dim,
              'testing': False}

    partition = {}
    
    if load_pt:
        partition = load_dict('cnn_data_partition.pkl')
    else:
        stack_IDs = os.listdir(im_path)
        random.shuffle(stack_IDs)
        
        train_IDs, test_IDs = np.split(stack_IDs, [int(len(stack_IDs)*0.95)])
        train_IDs, val_IDs = np.split(train_IDs, [int(len(train_IDs)*0.8)])
        
        # plot_input(im_path, train_IDs, subvolumes=subvolumes, img_size=target_size)
        
        partition['train'] = train_IDs
        partition['validation'] = val_IDs
        partition['testing'] = test_IDs
    
    train_generator = DataGenerator(partition['train'], **params)
    validation_generator = DataGenerator(partition['validation'], **params)

    return train_generator, validation_generator, partition

class DataGenerator(keras.utils.Sequence):
    
    """Generates data for Keras"""
    """This structure guarantees that the network will only train once on each sample per epoch"""

    def __init__(self, list_IDs, im_path, label_path, batch_size=4, 
                 img_dim=(128, 128, 128), dim=(256, 256, 256), n_channels=1, 
                 n_classes=2, shuffle=True, augment=False, subvolumes=False,
                 subvol_dim=None, testing=False):
        
        'Initialization'
        self.img_dim = img_dim
        self.dim = dim
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.im_path = im_path
        self.label_path = label_path
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.augment = augment
        self.subvolumes = subvolumes
        self.subvol_dim = subvol_dim
        self.on_epoch_end()
        self.testing = testing

        print('Found %d image stacks belonging to %d classes.' %
              (len(self.list_IDs), self.n_classes))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        if self.subvolumes:
            indexes = self.indexes[index * 1:(index + 1)]
        else:
            indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        
        X, Y = load_tensors(self, list_IDs_temp)

        if self.augment:
            X, Y = random_augmentation(X, Y)
                
        return X, Y
    
def load_tensors(self, list_IDs_temp):
    
    img_inVol = np.empty([self.batch_size, self.n_channels, self.dim[0], 
                                  self.dim[1], self.dim[2]], dtype='float32')
    img_maskVol = np.empty([self.batch_size, self.n_channels, self.dim[0], 
                            self.dim[1], self.dim[2]], dtype='float32')

    for i, ID in enumerate(list_IDs_temp):
        
        img_in_path = os.path.join(self.im_path, ID)
        img_mask_path = os.path.join(self.label_path, ID)
        
        if self.subvolumes:
            
            imgs_in = os.path.join(img_in_path, ID+"_input.npy")
            imgs_mask = os.path.join(img_mask_path, ID+"_mask.npy")
            img_inVol, img_maskVol = get_subVolumes(self, np.load(imgs_in), 
                                                    np.load(imgs_mask))
            
        else:
            
            imgs_in = os.path.join(img_in_path, ID+"_input.npy")
            imgs_mask = os.path.join(img_mask_path, ID+"_mask.npy")

            img_inVol[i, 0,] = np.load(imgs_in)
            img_maskVol[i, 0,] = np.load(imgs_mask)
        
        # if np.amin(img_maskVol) < 0.0:
        #     print('Error: input pixel value smaller than zero')
        # if np.amax(img_maskVol) > 1.0:
        #     print('Error: input pixel value greater than one')
    
    return img_inVol, img_maskVol

def get_subVolumes(self, input_stack, mask_stack, overlap_size=(0,0,0)):        
    
    img_inVol = np.empty([self.batch_size, self.n_channels, self.dim[0], 
                                  self.dim[1], self.dim[2]])
    img_maskVol = np.empty([self.batch_size, self.n_channels, self.dim[0], 
                            self.dim[1], self.dim[2]], dtype='uint8')
    batch_idx = 0
    subvol_size = self.subvol_dim
    zrange = np.arange(0, input_stack.shape[2], subvol_size[2]-overlap_size[2])
    zrange = zrange[1:3]
    for x in np.arange(0, input_stack.shape[0],
                       subvol_size[0]-overlap_size[0]):
        for y in np.arange(0, input_stack.shape[1],
                       subvol_size[1]-overlap_size[1]):
            for z in zrange:

                img_inVol[batch_idx, 0,] = input_stack[x:x+subvol_size[0],
                                                       y:y+subvol_size[1],
                                                       z:z+subvol_size[2]]
                
                img_maskVol[batch_idx, 0,] = mask_stack[x:x+subvol_size[0],
                                                        y:y+subvol_size[1],
                                                        z:z+subvol_size[2]]
                
                batch_idx += 1
                if batch_idx >= self.batch_size: break
            if batch_idx >= self.batch_size: break
        if batch_idx >= self.batch_size: break

    return img_inVol, img_maskVol
