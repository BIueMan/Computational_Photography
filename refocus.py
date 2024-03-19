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

    lensletSize = 16
    maxUV = (lensletSize - 1) / 2
    u = np.arange(1,u_max) - 1 - maxUV
    v = np.arange(1,v_max) - 1 - maxUV;

    sum_images = np.zeros_like(LF[0,0],dtype=np.float64)
    for ui, us in enumerate(u):
        for vi, vs in enumerate(v):
            sub_image = LF[ui,vi].astype(np.float64)

            interp_funcs = [interp2d(y, x, sub_image[:, :, i], kind='linear') for i in range(sub_image.shape[2])]
            inted_image = np.zeros_like(sub_image, dtype=np.float64)
            for i, interp_func in enumerate(interp_funcs):
                inted_image[:, :, i] = interp_func(y + vs*s, x + us*s)

            sum_images += mask[ui,vi]*inted_image

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

if __name__ == "__main__":
    # part 1
    path = 'data/chessboard_lightfield.png'
    image = cv2.imread(path)
    LF = arrangeLF(image, 16, 16)

    # part 2
    masks = read_mat()
    I = refocus(LF, -0.5, masks[0])
    plt.imshow(I/255)
    plt.show()