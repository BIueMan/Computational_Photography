import numpy as np
from lightfield_convorter import *
from read_mat import *
from refocus import *
import matplotlib.pyplot as plt
from all_focus import *

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
    start, end, step = 0, -1.6, -0.05
    I_id = []
    for m, mask in enumerate(masks):
        I = []
        for i in tqdm(np.arange(start, end, step)):
            I_i = refocus(LF, i, mask)
            I.append(I_i)
        I_print = [I[0], I[len(I)//2], I[-1]]
        I_id.append(np.array(I))

        ImArray2_j = np.concatenate(I_print, axis=1)
        plt.imsave(f'ex2_q2_m{m}.png', ImArray2_j/255)

    # part 3
    for idx, I in enumerate(I_id):
        gray_images = np.array([cv2.cvtColor(I_i.astype(np.uint8), cv2.COLOR_BGR2GRAY) for I_i in I]).astype(np.float64)
        w_sharpness = get_sharpness(gray_images, k=25, sigma1= 8, sigma2= 2)
        all_focus_image = calculate_all_focus(I, w_sharpness)
        depth = calculate_depth(w_sharpness)
        
        plt.imshow(all_focus_image/all_focus_image.max())
        plt.savefig(f'all_focus_m{idx}.png')

        plot_image_grid(w_sharpness, 8)
        plt.savefig(f'w_sharpness_grid_m{idx}.png')
        
        plt.imshow(depth/depth.max(), cmap='gray')
        plt.savefig(f'depth_m{idx}.png')
