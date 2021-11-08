'''
Implementation of V-Net architecture based on Milletari et al. (2016)
https://arxiv.org/pdf/1606.04797.pdf
'''
import keras.initializers as ki

from keras.models import Model
from keras.layers import (Input, concatenate, Conv3D, Conv3DTranspose)
from keras.optimizers import Adam
from keras.layers.advanced_activations import PReLU
from custom_loss import dice_coef_loss, dice_coef, jaccard_distance, surface_loss
from custom_activation import gelu
from func_utils import initializer_out

def optimiseCNN(train_gen, validation_gen, callbacks,
                targetsize=(128,128,128), batch_size=1, pool_size=(2,2,2), 
                n_labels=1):
    
    def optiVnet(dummyx, dummyy, dummyx_val, dummyy_val, params):
        
        history, model = Vnet(train_gen, validation_gen, callbacks=callbacks,
                                target_size=targetsize, batch_size=batch_size, 
                                n_labels=1, params=params)
        
        return history, model
        
    return optiVnet

def Vnet(train_gen, validation_gen, callbacks,
           target_size=(256,256,64), pool_size=(2,2,2), batch_size=4, 
           n_labels=1, params=[]):

    input_shape = (1, target_size[0], target_size[1], target_size[2])
    
    inputs = Input(shape = input_shape)
    
    acti = params['activation']
    ikernel = params['k_initializer']
    ds = params['ds']
    
    conv1, dconv = vnet_encoder(inputs=inputs, filters=int(64/ds), nlayers=1, 
                                 ikernel=ikernel, acti=acti)

    conv2, dconv = vnet_encoder(inputs=dconv, filters=int(128/ds), nlayers=2,
                                ikernel=ikernel, acti=acti)

    conv3, dconv = vnet_encoder(inputs=dconv, filters=int(256/ds), nlayers=3,
                                ikernel=ikernel, acti=acti)

    conv4, dconv = vnet_encoder(inputs=dconv, filters=int(512/ds), nlayers=3,
                                ikernel=ikernel, acti=acti)
    
    conv = vnet_encoder(inputs=dconv, filters=int(1024/ds), nlayers=3, 
                        ikernel=ikernel, acti=acti, dconv=False)

    conv = vnet_decoder(inputs=conv, conc=conv4, filters=int(512/ds), nlayers=3,
                        ikernel=ikernel, acti=acti)
    
    conv = vnet_decoder(inputs=conv, conc=conv3, filters=int(256/ds), nlayers=3,
                        ikernel=ikernel, acti=acti)
    
    conv = vnet_decoder(inputs=conv, conc=conv2, filters=int(128/ds), nlayers=2,
                        ikernel=ikernel, acti=acti)

    conv = vnet_decoder(inputs=conv, conc=conv1, filters=int(64/ds), nlayers=1,
                        ikernel=ikernel, acti=acti)
    
    conv = Conv3D(n_labels, (1, 1, 1), activation=params['flayer'], 
                  data_format='channels_first')(conv)

    model = Model(inputs=[inputs], outputs=[conv])
    
    model.compile(optimizer = Adam(lr = params['lr'], decay=params['decay']), loss = params['loss'], 
                  metrics = [dice_coef, jaccard_distance],
                  sample_weight_mode=None)
    
    history = model.fit_generator(train_gen,
                    steps_per_epoch = (target_size[2] // batch_size),
                    epochs = params['epochs'],
                    callbacks=callbacks,
                    validation_data = validation_gen,
                    validation_steps = len(validation_gen),
                    workers = 22,
                    use_multiprocessing=False,
                    max_queue_size = 4)

    return history, model


def vnet_encoder(inputs, filters, nlayers, ikernel=ki.he_uniform(seed=None), 
                 acti='prelu', dconv=True):
   
    for i in range(0,nlayers):    
        conv = Conv3D(filters, (5, 5, 5), strides=1, padding='same', 
                      kernel_initializer=ikernel, 
                      data_format='channels_first')(inputs)
        conv = PReLU()(conv)
    
    conv = concatenate([conv, inputs], axis=1)
    
    if dconv:
        dconv = Conv3D(filters, (2, 2, 2), strides=2, padding='same', 
                      kernel_initializer=ikernel, 
                      data_format='channels_first')(conv)
        dconv = PReLU()(dconv)
        return conv, dconv
    else:
        return conv


def vnet_decoder(inputs, conc, filters, nlayers, acti='prelu',
                 ikernel=ki.he_uniform(seed=None)):
    
    inputs = Conv3DTranspose(filters, (2, 2, 2), strides=2, padding='same', 
                             data_format='channels_first')(inputs)
    inputs = PReLU()(inputs)
    
    uconv = concatenate([inputs, conc], axis=1)
    
    for i in range(0,nlayers):    
        uconv = Conv3D(filters, (5, 5, 5), strides=1, padding='same', 
                      kernel_initializer=ikernel, 
                      data_format='channels_first')(uconv)
        uconv = PReLU()(uconv)
    
    uconv = concatenate([uconv, inputs], axis=1)
    
    return uconv