import numpy as np
from keras_preprocessing import image

def random_flip(img, mask, u=0.2):
    if np.random.random() < u:
        img = img[:,:,:,:,::-1]
        mask = mask[:,:,:,:,::-1]
    if np.random.random() < u:
        img = img[:,:,:,::-1]
        mask = mask[:,:,:,::-1]
    if np.random.random() < u:
        img = img[:,:,::-1]
        mask = mask[:,:,::-1]
    return img, mask

def random_rotate(img, mask, theta=np.random.uniform(-20, 20)):

    img = image.apply_affine_transform(img, theta=theta)
    mask = image.apply_affine_transform(mask, theta=theta)
        
    return img, mask

def shift(x, wshift, hshift, row_axis=1, col_axis=2):

    h, w = x.shape[row_axis], x.shape[col_axis]
    tx = hshift * h
    ty = wshift * w
    x = image.apply_affine_transform(x, ty=ty, tx=tx, channel_axis=1)

    return x

def random_shift(img, mask, wshift = np.random.uniform(-0.1,0.1),
                 hshift = np.random.uniform(-0.1,0.1)):
    
    img = shift(img, wshift, hshift)
    mask = shift(mask, wshift, hshift)
    
    return img, mask

# def random_zoom(img, mask, zoom_range=(0.8, 1), u=0.5):
#     if np.random.random() < u:
#         zx, zy = np.random.uniform(zoom_range[0], zoom_range[1], 2)
#         img = image.apply_affine_transform(img, zx=zx, zy=zy)
#         mask = image.apply_affine_transform(mask, zx=zx, zy=zy)
#     return img, mask

def random_shear(img, mask, intensity_range=(-0.5, 0.5),
                 sh = np.random.uniform(-0.5, 0.5)):

    img = image.apply_affine_transform(img, shear=sh, channel_axis=1)
    mask = image.apply_affine_transform(mask, shear=sh, channel_axis=1)
        
    return img, mask

def random_contrast(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        gray = img
        gray = (3.0 * (1.0 - alpha) / gray.size) * np.sum(gray)
        img = alpha * img + gray
        img = np.clip(img, 0., 1.)
    return img

def random_brightness(img, limit=(-0.3, 0.3), u=0.5):
    if np.random.random() < u:
        alpha = 1.0 + np.random.uniform(limit[0], limit[1])
        img = alpha * img
        img = np.clip(img, 0., 1.)
    return img

# def random_saturation(img, limit=(-0.3, 0.3), u=0.5):
#     if np.random.random() < u:
#         alpha = 1.0 + np.random.uniform(limit[0], limit[1])
#         coef = np.array([[[0.114, 0.587, 0.299]]])
#         gray = img * coef
#         gray = np.sum(gray, axis=2, keepdims=True)
#         img = alpha * img + (1. - alpha) * gray
#         img = np.clip(img, 0., 1.)
#     return img

# def random_channel_shift(x, limit, channel_axis=2):
#     x = np.rollaxis(x, channel_axis, 0)
#     min_x, max_x = np.min(x), np.max(x)
#     channel_images = [np.clip(x_ch + np.random.uniform(-limit, limit), min_x, max_x) for x_ch in x]
#     x = np.stack(channel_images, axis=0)
#     x = np.rollaxis(x, 0, channel_axis + 1)
#     return x

# def random_crop(img, mask, u=0.1):
#     if np.random.random() < u:
#         w, h = img.shape[0], img.shape[1]
#         offsetw = np.random.randint(w//2)
#         offseth = np.random.randint(w//2)

#         endw = np.random.randint(w // 2)+w // 2
#         endh = np.random.randint(w // 2)+w // 2

#         new_im = img[offsetw:offsetw + endw, offseth:offseth + endh, :]
#         new_mask = mask[offsetw:offsetw + endw, offseth:offseth + endh, :]

#         new_im, new_mask = cv2.resize(new_im, interpolation = cv2.INTER_LINEAR, dsize=(w, h)), \
#                cv2.resize(new_mask, interpolation=cv2.INTER_LINEAR, dsize=(w, h))

#         new_mask = new_mask[..., np.newaxis]
#         return new_im, new_mask
#     else:
#         return img, mask


def random_augmentation(img, mask):
    
    # for i in range(0, img.shape[0]):
    #     #  Assuming isotropic volume
    #     intensity_range=(-0.5, 0.5)
    #     shx = np.random.uniform(-intensity_range[0], intensity_range[1])
    #     shy = np.random.uniform(-intensity_range[0], intensity_range[1])
    #     shz = np.random.uniform(-intensity_range[0], intensity_range[1])
        
    #     w_limit=(-0.1, 0.1)
    #     h_limit=(-0.1, 0.1)
    #     wshift = np.random.uniform(w_limit[0], w_limit[1])
    #     hshift = np.random.uniform(h_limit[0], h_limit[1])
    #     dshift = np.random.uniform(h_limit[0], h_limit[1])
        
    #     rotate_limit=(-20, 20)
    #     theta1 = np.random.uniform(rotate_limit[0], rotate_limit[1])
    #     theta2 = np.random.uniform(rotate_limit[0], rotate_limit[1])
    #     theta3 = np.random.uniform(rotate_limit[0], rotate_limit[1])
        
    #     # Random 2D shearing
    #     if np.random.random() < 0.05:
    #         for j in range(0,img.shape[4]):
    #             img[i,:,:,:,j], mask[i,:,:,:,j] = random_shear(img[i,:,:,:,j], 
    #                                                             mask[i,:,:,:,j], 
    #                                                             sh=shz)
    #     if np.random.random() < 0.05:
    #         for j in range(0,img.shape[4]):
    #             img[i,:,:,j,:], mask[i,:,:,j,:] = random_shear(img[i,:,:,j,:], 
    #                                                             mask[i,:,:,j,:], 
    #                                                             sh=shy)
    #     if np.random.random() < 0.05:
    #         for j in range(0,img.shape[4]):
    #             img[i,:,j,:,:], mask[i,:,j,:,:] = random_shear(img[i,:,j,:,:], 
    #                                                             mask[i,:,j,:,:], 
    #                                                             sh=shx)
                
    #     # Random 2D shifting
    #     if np.random.random() < 0.05:
    #         for j in range(0,img.shape[4]):
    #             img[i,:,:,:,j], mask[i,:,:,:,j] = random_shift(img[i,:,:,:,j], 
    #                                                             mask[i,:,:,:,j],
    #                                                             wshift=wshift,
    #                                                             hshift=hshift)
    #     if np.random.random() < 0.05:
    #         for j in range(0,img.shape[4]):
    #             img[i,:,:,j,:], mask[i,:,:,j,:] = random_shift(img[i,:,:,j,:], 
    #                                                             mask[i,:,:,j,:],
    #                                                             wshift=wshift,
    #                                                             hshift=dshift)
    #     if np.random.random() < 0.05:
    #         for j in range(0,img.shape[4]):
    #             img[i,:,j,:,:], mask[i,:,j,:,:] = random_shift(img[i,:,j,:,:], 
    #                                                             mask[i,:,j,:,:],
    #                                                             wshift=hshift,
    #                                                             hshift=dshift)
                
    #     # Random 2D rotation
    #     if np.random.random() < 0.05:
    #         for j in range(0,img.shape[4]):
    #             img[i,:,:,:,j], mask[i,:,:,:,j] = random_rotate(img[i,:,:,:,j], 
    #                                                             mask[i,:,:,:,j],
    #                                                             theta=theta3)
    #     if np.random.random() < 0.05:
    #         for j in range(0,img.shape[4]):
    #             img[i,:,:,j,:], mask[i,:,:,j,:] = random_rotate(img[i,:,:,j,:], 
    #                                                             mask[i,:,:,j,:],
    #                                                             theta=theta2)
    #     if np.random.random() < 0.05:
    #         for j in range(0,img.shape[4]):
    #             img[i,:,j,:,:], mask[i,:,j,:,:] = random_rotate(img[i,:,j,:,:], 
    #                                                             mask[i,:,j,:,:],
    #                                                             theta=theta1)
                
    
    # Transform entire 5D stack
    img = random_brightness(img, limit=(-0.1, 0.1), u=0.5)
    img = random_contrast(img, limit=(-0.1, 0.1), u=0.5)
    img, mask = random_flip(img, mask)
    
    # img = random_saturation(img, limit=(-0.1, 0.1), u=0.05)
    # img, mask = random_zoom(img, mask, zoom_range=(0.9, 1.1), u=0.05)
    
    return img, mask