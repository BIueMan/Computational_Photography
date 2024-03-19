import numpy as np
import cv2
from lightfield_convorter import *
from read_mat import *
from refocus import *
from tqdm import tqdm

def get_grey_arrangeLF(arrangeLF):
    U,V,S,T,C = arrangeLF.shape
    
    grey_LF = np.zeros([U,V,S,T])
    for u in range(U):
        for v in range(V):
            grey_LF[u,v] = cv2.cvtColor(LF[u,v], cv2.COLOR_BGR2GRAY)
            
    return grey_LF
            

def get_kernel_2d(k:int, sigma:float) -> np.ndarray:
    # Create the Gaussian kernel
    kernel = cv2.getGaussianKernel(k, sigma)
    kernel_2d = np.outer(kernel, kernel.T)
    # normalize the kernel
    kernel_2d = kernel_2d/np.sum(kernel_2d)
    return kernel_2d

def get_sharpness(gray_image_list, k, sigma1, sigma2):
    kernel_1 = get_kernel_2d(k, sigma1)
    kernel_2 = get_kernel_2d(k, sigma2)
    
    sharpness_list = []
    for gray_image in gray_image_list:
        blurred_image = cv2.filter2D(gray_image, -1, kernel_1)
        high_freq = gray_image - blurred_image
        sharpness_list.append(cv2.filter2D(high_freq*high_freq, -1, kernel_2))    
    
    w_sharpness = np.array(sharpness_list)
    return w_sharpness
    

def calculate_all_focus(I, w_sharpness):
    # sum(w_sharpness * I)
    numerator = np.sum(w_sharpness[..., np.newaxis] * I, axis=0)
    # sum(w_sharpness)
    denominator = np.sum(w_sharpness, axis=0)

    # Calculate the all-focus image
    all_focus_image = numerator / denominator[..., np.newaxis]
    
    return all_focus_image

def calculate_depth(w_sharpness):
    # sum(w_sharpness * d)
    D = np.arange(w_sharpness.shape[0])
    numerator = np.sum(w_sharpness * D[..., np.newaxis,np.newaxis], axis=0)
    # sum(w_sharpness)
    denominator = np.sum(w_sharpness, axis=0)

    # Calculate the depth map
    depth_map = numerator / denominator
    
    return depth_map

def plot_image_grid(images, n_columns=8):
    num_images, image_height, image_width = images.shape  # Get image dimensions

    # Calculate total width of the grid
    total_width = n_columns * image_width
    total_height = (num_images + n_columns - 1) // n_columns * image_height

    fig, axes = plt.subplots(num_images // n_columns, n_columns, figsize=(total_width / 200, total_height / 200))

    for i in range(num_images):
        row_index = i // n_columns
        col_index = i % n_columns
        ax = axes[row_index, col_index]
        ax.imshow(images[i])
        ax.set_title(f"Image {i+1}")
        ax.axis('off')

    for j in range(num_images, (num_images // n_columns ) * n_columns):  # Clearing extra subplots
        row_index = j // n_columns
        col_index = j % n_columns
        fig.delaxes(axes[row_index, col_index])

    plt.tight_layout()

if __name__ == "__main__":
    # part 1
    path = 'data/chessboard_lightfield.png'
    image = cv2.imread(path)
    LF = arrangeLF(image, 16, 16)
    ImArray1 = combine_to_big(LF)

    # part 2
    masks = read_mat()
    start, end, step = 0, -1.6, -0.05
    I_id = []
    
    mask = masks[0]
    I = []
    for i in tqdm(np.arange(start, end, step)):
        I_i = refocus(LF, i, mask)
        I.append(I_i)
    I = np.array(I)

    gray_image = np.array([cv2.cvtColor(I_i.astype(np.uint8), cv2.COLOR_BGR2GRAY) for I_i in I]).astype(np.float64)
    w_sharpness = get_sharpness(gray_image, k=25, sigma1= 8, sigma2= 2)
    all_focus_image = calculate_all_focus(I, w_sharpness)
    depth = calculate_depth(w_sharpness)
    
    plt.imshow(all_focus_image/all_focus_image.max())
    plt.show()

    plot_image_grid(w_sharpness, 8)
    plt.show()
    
    plt.imshow(depth/depth.max(), cmap='gray')
    plt.show()
