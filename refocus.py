import numpy as np
from fractions import Fraction
from scipy.interpolate import interp2d
import matplotlib.pyplot as plt
import cv2
from lightfield_convorter import *
from read_mat import *
from tqdm import tqdm

def refocus(LF, s, mask):
    u_max, v_max, s_max, t_max, c = LF.shape

    # Extend the image by padding
    x = np.arange(0, s_max, 1)
    y = np.arange(0, t_max, 1)

    sum_images = np.zeros_like(LF[0,0],dtype=np.float64)
    for u in range(u_max):
        for v in range(v_max):
            sub_image = LF[u,v].astype(np.float64)

            interp_funcs = [interp2d(y, x, sub_image[:, :, i], kind='linear') for i in range(sub_image.shape[2])]
            inted_image = np.zeros_like(sub_image, dtype=np.float64)
            for i, interp_func in enumerate(interp_funcs):
                inted_image[:, :, i] = interp_func(y + v*s, x + u*s)

            sum_images += mask[u,v]*inted_image

    return sum_images

def split_fraction(fraction_float):
    fraction_str = str(fraction_float)
    fraction = Fraction(fraction_str)
    return fraction.numerator, fraction.denominator

def pad_image(image):
    h, w, c = image.shape
    p = np.zeros((h + 1, w + 1, c), dtype=image.dtype)
    p[:-1, :-1, :] = image
    p[-1, :-1, :] = image[-1, :, :]
    p[:-1, -1, :] = image[:, -1, :]
    p[-1, -1, :] = image[-1, -1, :]
    return p

# def bilinear_interpolation(image, su, sv):
#     # full roll
#     image.roll(np.floor(su).astype(int), axis = 0)
#     image.roll(np.floor(sv).astype(int), axis = 1)
#     # keep left over
#     x = su%1
#     y = sv%1

#     x1, y1 = np.floor(x).astype(int), np.floor(y).astype(int)
#     x2, y2 = x1 + 1, y1 + 1

#     # Clip coordinates to stay within image bounds
#     x1 = np.clip(x1, 0, image.shape[1] - 1)
#     y1 = np.clip(y1, 0, image.shape[0] - 1)
#     x2 = np.clip(x2, 0, image.shape[1] - 1)
#     y2 = np.clip(y2, 0, image.shape[0] - 1)

#     # Calculate the weights
#     wa = (x2 - x) * (y2 - y)
#     wb = (x - x1) * (y2 - y)
#     wc = (x2 - x) * (y - y1)
#     wd = (x - x1) * (y - y1)

#     # Perform bilinear interpolation
#     interpolated_value = (wa * image+ 
#                           wb * image.roll(1, axis = 0) + 
#                           wc * image.roll(1, axis = 1) + 
#                           wd * image.roll(1, axis = 0).roll(1, axis = 1))

#     return interpolated_value

if __name__ == "__main__":
    # part 1
    path = 'data/chessboard_lightfield.png'
    image = cv2.imread(path)
    LF = arrangeLF(image, 16, 16)

    # part 2
    masks = read_mat()
    I = refocus(LF, 0, masks[0])
    plt.imshow(I/255)
    plt.show()