import numpy as np
import os

img_matrix = np.load(os.path.join(os.getcwd(), 'test_data.npy'))
print(img_matrix.shape)