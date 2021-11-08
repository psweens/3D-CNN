import tensorflow as tf
import numpy as np
from keras import backend as K
from scipy.ndimage import distance_transform_edt as distance


def dice_coef(y_true, y_pred, smooth=1):
    y_true_f = K.batch_flatten(y_true)
    y_pred_f = K.batch_flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f, axis=1)
    return (2. * intersection + smooth) / (K.sum(y_true_f, axis=1) + K.sum(y_pred_f, axis=1) + smooth)

def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)

def msq(y_true, y_pred):
    return K.sum(K.pow(y_true - y_pred, 2), axis=None)

def msq_loss(y_true, y_pred):
    return msq(y_true, y_pred)

# https://stackoverflow.com/questions/43033436/how-to-do-point-wise-categorical-crossentropy-loss-in-keras
def image_softmax(input):  # apply softmax activation to a 4D tensor
    label_dim = 1
    d = K.exp(input - K.max(input, axis=label_dim, keepdims=True))
    return d / K.sum(d, axis=label_dim, keepdims=True)

def image_categorical_crossentropy(y_true, y_pred):  # compute cross-entropy on 4D tensors
    y_pred = K.clip(y_pred,  1e-5, 1 -  1e-5)
    return -K.mean(y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))

def image_categorical_crossentropy_loss(y_true, y_pred):  # compute cross-entropy on 4D tensors
    return 1 - image_categorical_crossentropy(y_true, y_pred)

# def dice_coef(y_true, y_pred, smooth=1.):
#     y_true_f = K.flatten(y_true)
#     y_pred_f = K.flatten(y_pred)
#     intersection = K.sum(y_true_f * y_pred_f)
#     return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)



# def neg_dice_coef_loss(y_true, y_pred):
#     return -neg_dice_coef(y_true, y_pred)


def soft_dice(y_pred, y_true):
    # y_pred is softmax output of shape (num_samples, num_classes)
    # y_true is one hot encoding of target (shape= (num_samples, num_classes))
    intersect = K.sum(y_pred * y_true, 0)
    denominator = K.sum(y_pred, 0) + K.sum(y_true, 0)
    dice_scores = K.constant(2) * intersect / (denominator + K.constant(1e-6))
    return dice_scores


def jaccard_distance(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation.
    Also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if
    each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat.
    If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy)
    or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.
    # Returns
        The Jaccard distance between the two tensors.
    # References
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    sum_ = K.sum(K.abs(y_true) + K.abs(y_pred), axis=-1)
    jac = (intersection + smooth) / (sum_ - intersection + smooth)
    return (1 - jac) * smooth

'''
Surface Loss code
'''
def calc_dist_map(seg):
    res = np.zeros_like(seg)
    posmask = seg.astype(np.bool)

    if posmask.any():
        negmask = ~posmask
        res = distance(negmask) * negmask - (distance(posmask) - 1) * posmask

    return res

def calc_dist_map_batch(y_true):
    y_true_numpy = y_true.numpy()
    return np.array([calc_dist_map(y)
                     for y in y_true_numpy]).astype(np.float32)

def surface_loss(y_true, y_pred):
    y_true_dist_map = tf.py_function(func=calc_dist_map_batch,
                                     inp=[y_true],
                                     Tout=tf.float32)
    multipled = y_pred * y_true_dist_map
    return K.mean(multipled)