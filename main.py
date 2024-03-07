import numpy as np
from lightfield_convorter import *
from read_mat import *
from refocus import *
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # part 1
    path = 'data/chessboard_lightfield.png'
    image = cv2.imread(path)
    LF = arrangeLF(image, 16, 16)
    ImArray1 = combine_to_big(LF)
    # save image
    plt.imsave('ex1_q1.png', ImArray1/255)

    # part 2
    masks = read_mat()
    I_0 = refocus(LF, 0, masks[0])
    I_1 = refocus(LF, -0.5, masks[0])
    I_2 = refocus(LF, -1, masks[0])

    combined_image = np.concatenate((I_0, I_1, I_2), axis=1)
    plt.imsave('ex2_q2.png', combined_image/255)
