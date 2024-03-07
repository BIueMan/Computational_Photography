import numpy as np
from scipy.io import loadmat

def read_mat(path = 'data/masks.mat'):
    masks = loadmat(path)
    masks = [masks[f'mask{idx}'] for idx in range(1,5)]

    return masks

if __name__ == "__main__":
    masks = read_mat()
    print(masks)
