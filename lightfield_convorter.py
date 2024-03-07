import numpy as np
import cv2

import matplotlib.pyplot as plt

def arrangeLF(I, u_max, v_max):
    u_blocks = I.shape[0] // u_max
    v_blocks = I.shape[1] // v_max

    blocks = I[:u_blocks*u_max, :v_blocks*v_max].reshape(u_blocks, u_max, v_blocks, v_max, I.shape[2])

    LF = blocks.transpose(1, 3, 0, 2, 4).reshape(u_max, v_max, u_blocks, v_blocks, I.shape[2])

    return LF

def combine_to_big(LF):
    u_max, v_max, u_blocks, v_blocks, channels = LF.shape

    big_image = LF.transpose(0, 2, 1, 3, 4).reshape(u_max * u_blocks, v_max * v_blocks, channels)

    return big_image


if __name__ == "__main__":
    path = 'data/chessboard_lightfield.png'
    image = cv2.imread(path)
    lightfield_images = arrangeLF(image, 16, 16)
    # test 1 small image
    plt.imshow(lightfield_images[0,0])
    plt.title("lightfield_images[0,0]")
    plt.show()

    ImArray1 = combine_to_big(lightfield_images)
    # test part image
    plt.imshow(ImArray1[0:lightfield_images.shape[2],0:lightfield_images.shape[3]])
    plt.title("ImArray1[0,0]")
    plt.show()

    # save image
    plt.imsave('ex1_q1.png', ImArray1/255)