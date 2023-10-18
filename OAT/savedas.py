import os
import numpy as np
from torchvision import utils as vutils
import torch

def gettestdata(cache_dir, n_name):
    
    print('Obtaining data for testing...')
    Y = np.load(os.path.join(cache_dir, 'Y'+n_name+'.npy')) # True image
    SNR = np.load(os.path.join(cache_dir, 'SNR'+n_name+'.npy')) # True image
    Y=Y.astype(np.float32)
    SNR=SNR.astype(np.float32)
    
    Xdas = np.load(os.path.join(cache_dir, 'Xdas'+n_name+'.npy')) # Noisy image obtained with DAS
    Xdas=Xdas.astype(np.float32)
    return Xdas,Y,SNR


def save_imgs(X, dataset):
    save_dir = dataset + '_images'
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(50):
        # Convert the predicted image to a PyTorch tensor and scale to [0, 1]
        predicted_image = X[i, :, :]
        predicted_image = torch.tensor(predicted_image, dtype=torch.float32)        
        #predicted_image = (predicted_image - torch.min(predicted_image)) / (torch.max(predicted_image) - torch.min(predicted_image))

        # Use vutils to save the image
        save_path = os.path.join(save_dir, dataset+f'_image_{i}.png')
        vutils.save_image(predicted_image, save_path, normalize=True)
        print('images generated')
    

if __name__ == '__main__':
    
    cache_dir = 'data/' 
    datadate = '9aug23'
    nname = 'a' + datadate # name set for testing
    Xdas,_,_ = gettestdata(cache_dir, nname)
    save_imgs(Xdas, "das")