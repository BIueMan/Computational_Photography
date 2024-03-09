import numpy as np
import cv2
from lightfield_convorter import *
from read_mat import *

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
    
    sharpness_list = np.zeros_like(gray_image_list)
    for u in range(sharpness_list.shape[0]):
        for v in range(sharpness_list.shape[1]):
            gray_image = gray_image_list[u,v]
            
            blurred_image = cv2.filter2D(gray_image, -1, kernel_1)
            high_freq = gray_image - blurred_image
            sharpness_list[u,v] = cv2.filter2D(high_freq*high_freq, -1, kernel_2)
            
            
    return sharpness_list
    

def calculate_all_focus(I, w_sharpness):
    # sum(w_sharpness * I)
    numerator = np.sum(w_sharpness[..., np.newaxis] * I, axis=(0,1))
    # sum(w_sharpness)
    denominator = np.sum(w_sharpness, axis=(0,1))

    # Calculate the all-focus image
    all_focus_image = numerator / denominator[..., np.newaxis]
    
    return all_focus_image

def calculate_depth(w_sharpness):
    # sum(w_sharpness * d)
    D = np.fromfunction(lambda i, j: (i+1)*(j+1), w_sharpness.shape[:2])
    numerator = np.sum(w_sharpness * D[..., np.newaxis,np.newaxis], axis=(0,1))
    # sum(w_sharpness)
    denominator = np.sum(w_sharpness, axis=(0,1))

    # Calculate the depth map
    depth_map = numerator / denominator
    
    return depth_map


if __name__ == "__main__":
    path = 'data/chessboard_lightfield.png'
    image = cv2.imread(path)
    LF = arrangeLF(image, 16, 16)
    gray_image = get_grey_arrangeLF(LF)
    
    w_sharpness = get_sharpness(gray_image, k=5, sigma1= 1, sigma2= 2)
    all_focus_image = calculate_all_focus(LF, w_sharpness)
    depth = calculate_depth(w_sharpness)
    
    plt.imshow(all_focus_image/all_focus_image.max())
    plt.show()
    
    plt.imshow(depth, cmap='gray')
    plt.show()
