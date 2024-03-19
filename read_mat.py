import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt

def read_mat(path = 'data/masks.mat'):
    masks = loadmat(path)
    masks = [masks[f'mask{idx}'] for idx in range(1,5)]

    return masks

if __name__ == "__main__":
    masks = read_mat()
    def plot_image_grid(images):
        num_rows = 2
        num_columns = 2

        fig, axes = plt.subplots(num_rows, num_columns, figsize=(8, 8))

        for i in range(num_rows * num_columns):
            row_index = i // num_columns
            col_index = i % num_columns
            ax = axes[row_index, col_index]
            ax.imshow(images[i], cmap='gray', vmin=0, vmax=0.05)  # Set color limits
            ax.set_title(f"Masks {i}")
            ax.axis('off')

        plt.tight_layout()
        return plt
    plot_image_grid(np.array(masks))
    plt.show()
