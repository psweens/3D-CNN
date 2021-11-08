import keras_metrics as km
import tensorflow as tf
import keras.initializers as ki
import tensorflow.keras
from keras.models import Model
from keras.layers import (Input, concatenate, Dropout, Conv3D, MaxPooling3D,
                          Conv3DTranspose, BatchNormalization, Activation,
                          LeakyReLU, AlphaDropout)
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping
from keras.utils import get_custom_objects
from custom_loss import dice_coef_loss, dice_coef, jaccard_distance, soft_dice
from custom_activation import gelu
from func_utils import initializer_out

def optimiseCNN(train_gen, validation_gen, callbacks,
                targetsize=(128,128,128), batch_size=1, pool_size=(2,2,2),
                n_labels=1):

    def optiUnet3D(dummyx, dummyy, dummyx_val, dummyy_val, params):

        history, model = Unet3D(train_gen, validation_gen, callbacks=callbacks,
                                target_size=targetsize, batch_size=batch_size,
                                n_labels=1, params=params)

        return history, model

    return optiUnet3D

def Unet3D(train_gen, validation_gen, callbacks,
           target_size=(256,256,64), pool_size=(2,2,2), batch_size=4,
           n_labels=1, training_size=None, params=[]):
    """
    Builds the 3D UNet Keras model.
    :param input_shape: Shape of the input data (n_channels, x_size, y_size, z_size).
    :param ds: Factor to which to reduce the number of filters. Making this value larger will
    reduce the amount of memory the model will need during training.
    :param pool_size: Pool size for the max pooling operations.
    :param n_labels: Number of binary labels that the model is learning.
    :param initial_learning_rate: Initial learning rate for the model. This will be decayed during training.
    :param deconvolution: If set to True, will use transpose convolution(deconvolution) instead of upsamping. This
    increases the amount memory required during training.
    :return: Untrained 3D UNet Model

    """
    input_shape = (1, target_size[0], target_size[1], target_size[2])

    inputs = Input(shape = input_shape)

    acti = params['activation']
    drop_fac = params['dropout']
    ikernel = params['k_initializer'] #initializer_out(params['k_initializer'], seed=None)
    ds = params['ds']

    conv1 = unet_encoder(inputs, filters=int(64/ds), ikernel=ikernel, acti=acti)
    conv = MaxPooling3D(pool_size=(2,2,2), strides=2, padding='same', 
                 data_format='channels_first')(conv1)

    conv2 = unet_encoder(conv, filters=int(128/ds), ikernel=ikernel, acti=acti)
    conv = MaxPooling3D(pool_size=(2,2,2), strides=2, padding='same', 
                 data_format='channels_first')(conv2)

    conv3 = unet_encoder(conv, filters=int(256/ds), ikernel=ikernel, acti=acti)
    conv = Dropout(drop_fac)(conv3)
    conv = MaxPooling3D(pool_size=(2,2,2), strides=2, padding='same', 
                 data_format='channels_first')(conv)

    conv4 = unet_encoder(conv, filters=int(512/ds), ikernel=ikernel, acti=acti)
    conv = Dropout(drop_fac)(conv4)
    conv = MaxPooling3D(pool_size=(2,2,2), strides=2, padding='same', 
                 data_format='channels_first')(conv)

    conv = unet_encoder(conv, filters=int(1028/ds), ikernel=ikernel, acti=acti)
    conv = Dropout(drop_fac)(conv)

    conv = unet_decoder(conv, conv4, filters=int(512/ds), ikernel=ikernel,
                          acti=acti)

    conv = unet_decoder(conv, conv3, filters=int(256/ds), ikernel=ikernel,
                         acti=acti)

    conv = unet_decoder(conv, conv2, filters=int(128/ds), ikernel=ikernel,
                         acti=acti)

    conv = unet_decoder(conv, conv1, filters=int(64/ds), ikernel=ikernel,
                         acti=acti)
    
    conv = Conv3D(n_labels, (1, 1, 1), activation=params['flayer'], data_format='channels_first')(conv)

    model = Model(inputs=[inputs], outputs=[conv])

    model.compile(optimizer = Adam(lr = params['lr'], decay=params['decay']), loss = params['loss'],
                  metrics = [dice_coef, jaccard_distance],
                  sample_weight_mode=None)

    history = model.fit_generator(train_gen,
                    steps_per_epoch = (training_size // batch_size),
                    epochs = params['epochs'],
                    callbacks=callbacks,
                    validation_data = validation_gen,
                    validation_steps = len(validation_gen),
                    workers = 22, 
                    use_multiprocessing=False,
                    max_queue_size = 4)

    return history, model


def unet_encoder(conv, filters, ikernel=ki.he_uniform(seed=None), acti='relu',
                 kernel_size=(3,3,3), nlayers=2):

    # inputs = conv
    for i in range(nlayers):
        conv = Conv3D(filters, kernel_size, strides=1, padding='same', 
                      data_format='channels_first',
                      activation=acti,
                      kernel_initializer=ikernel)(conv)
    # conv = concatenate([conv, inputs], axis=1)
        
    return conv


def unet_decoder(dconv, conc, filters, kernel_size=(3,3,3), nlayers=2, 
                 ikernel=ki.he_uniform(seed=None), acti='relu'):

    dconv = concatenate([Conv3DTranspose(filters, (2, 2, 2), strides=2,
                                       padding='same', data_format='channels_first')(dconv), conc], axis=1)
    
    for i in range(nlayers):
        dconv = Conv3D(filters, kernel_size, strides=1, padding='same', 
                      data_format='channels_first',
                      activation=acti,
                      kernel_initializer=ikernel)(dconv)

    # dconv = Conv3D(filters, (3, 3, 3), padding='same', kernel_initializer=ikernel,
    #                data_format='channels_first')(dconv)
    # dconv = custom_activation(input_tensor=dconv, activate=acti)

    # dconv = Conv3D(filters, (3, 3, 3), padding='same', kernel_initializer=ikernel,
    #                data_format='channels_first')(dconv)
    # dconv = custom_activation(input_tensor=dconv, activate=acti)

    return dconv


def custom_activation(input_tensor, activate, bnacti='relu'):

    if activate == 'relu':
        atensor = Activation('relu')(input_tensor)
    elif activate == 'elu':
        atensor = Activation('elu')(input_tensor)
    elif activate == 'bn':
        atensor = BatchNormalization(axis=1,momentum=0.5)(input_tensor)
        atensor = Activation(bnacti)(input_tensor)
    elif activate == 'leakyrelu':
        atensor = LeakyReLU(alpha=0.3)(input_tensor)
    elif activate == 'gelu':
        get_custom_objects().update({'gelu': Activation(gelu)})
        atensor = Activation(gelu)(input_tensor)
    elif activate == 'selu':
        atensor = Activation('selu')(input_tensor)
    

    return atensor