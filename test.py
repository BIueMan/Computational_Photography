import numpy as np
from scipy.interpolate import interp2d

# Sample data points
x = np.linspace(0, 4, 5)
y = np.linspace(0, 4, 5)
# Sample color image (3 channels: Red, Green, Blue)
z = np.random.rand(5, 5, 3)  # Random data for demonstration

# Create the interpolation function for each color channel
interp_funcs = [interp2d(x, y, z[:, :, i], kind='linear') for i in range(z.shape[2])]

# Define new points for interpolation
x_new = np.linspace(0, 4, 20+1)
y_new = np.linspace(0, 4, 20+1)

# Perform interpolation for each color channel
z_new = np.zeros((len(x_new), len(y_new), z.shape[2]))
for i, interp_func in enumerate(interp_funcs):
    z_new[:, :, i] = interp_func(x_new, y_new)

print("Original Image Shape:", z.shape)
print("Interpolated Image Shape:", z_new.shape)
