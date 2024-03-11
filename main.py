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
    start, end, step = 0, -1.2, -0.05
    I_id = []
    for m, mask in enumerate(masks):
        I = []
        for i in tqdm(np.arange(start, end, step)):
            I_i = refocus(LF, i, mask)
            I.append(I_i)
        I_print = [I[0], I[len(I)//2], I[-1]]
        I_id.append(I_i)

        combined_image = np.concatenate(I_print, axis=1)
        plt.imsave(f'ex2_q2_m{m}.png', combined_image/255)
