from dataclasses import dataclass

import numpy as np
import os

from fdunetln import FDUNet
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from quality import FoM
from tqdm import tqdm
from torchvision import utils as vutils

# ---------------------------------------------------------------------------
@dataclass
class TestingConfig:
    
    image_size = 128  # the image size and shape
    test_batch_size = 16
    device = 'cpu'

    cache_dir = 'data/' 
    datadate = '1aug23'
    nname = 't' + datadate # name set for testing
    
    traindate = '9aug23' 
    
    ckp_last='fdunet' + traindate + '.pth' # name of the file of the saved weights of the trained net
    ckp_best='fdunet_best' + traindate + '.pth'
    
    logfilename = 'TrainingLog_FDUNet' + traindate + '.log'
    
    dasorlbp = 'das' #lbp

# ---------------------------------------------------------------------------
def gettestdata(cache_dir, n_name, dasorlbp):
    
    print('Obtaining data for testing...')
    Y = np.load(os.path.join(cache_dir, 'Y'+n_name+'.npy')) # True image
    SNR = np.load(os.path.join(cache_dir, 'SNR'+n_name+'.npy')) # True image
    Y=Y.astype(np.float32)
    SNR=SNR.astype(np.float32)
    
    if dasorlbp == 'das':
        Xdas = np.load(os.path.join(cache_dir, 'Xdas'+n_name+'.npy')) # Noisy image obtained with DAS
        Xdas=Xdas.astype(np.float32)
        return Xdas,Y,SNR
    else:
        Xlbp = np.load(os.path.join(cache_dir, 'Xlbp'+n_name+'.npy')) # Noisy image obtained with LBP
        Xlbp=Xlbp.astype(np.float32)
        return Xlbp,Y,SNR

# ---------------------------------------------------------------------------
class OAImageDataset(Dataset):
    def __init__(self, X, Y):
        super(OAImageDataset, self).__init__()
        self.X = X
        self.Y = Y

    def __getitem__(self, item):
        return self.X[item, :, :], self.Y[item, :, :]

    def __len__(self):
        return self.X.shape[0]


# ---------------------------------------------------------------------------
def get_testloader(X, Y, batch_size): 
        
    dataset_test = OAImageDataset(X, Y)
    
    # Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=True) 
    test_loader = DataLoader(dataset_test, shuffle=True,  drop_last=False, **loader_args)
    
    return test_loader


# ---------------------------------------------------------------------------

def save_imgs(X, dataset):
    save_dir = dataset + '_images'
    os.makedirs(save_dir, exist_ok=True)
    
    for i in range(50):
        # Convert the predicted image to a PyTorch tensor and scale to [0, 1]
        predicted_image = X[i, :, :]
        predicted_image = torch.tensor(predicted_image, dtype=torch.float32)        
        predicted_image = (predicted_image - torch.min(predicted_image)) / (torch.max(predicted_image) - torch.min(predicted_image))

        # Use vutils to save the image
        save_path = os.path.join(save_dir, dataset+f'_image_{i}.png')
        vutils.save_image(predicted_image.add(1).mul(0.5), save_path, normalize=True)
        print('images generated')
        
# ---------------------------------------------------------------------------       

def predict():
    
    config = TestingConfig()
    
    # Create net object
    net = FDUNet(config.image_size,config.image_size).to(device=config.device)
    # Loading best checkpoint
    checkpoint = torch.load(config.ckp_best,map_location=torch.device(config.device))
    net.load_state_dict(checkpoint['state_dict'])
    
    # Load dataset
    Xi,Y,SNR = gettestdata(config.cache_dir, config.nname, config.dasorlbp)
       
    # Create data loader
    X = torch.as_tensor(Xi).type(torch.float32) 
    
    # Prediction
    print('Making prediction...')
    with torch.no_grad():
        X=X.to(device=config.device) 
        X = torch.unsqueeze(X,1)
        X = X.type(torch.float32)
        # net prediction
        pred = net.forward(X)
        pred = torch.squeeze(pred,1)
    
    print('done!')
    Ynet = pred.detach().to("cpu").numpy()
    
    B,H,W = Y.shape
    
    save_imgs(X, "reconstructed")
    save_imgs(Y, "orig")
    save_imgs(Ynet, "predicted")
    
    
    # Measuring the quality of the reconstruction 
    print('Calculating metrics...')
    SSIM = np.zeros((B,2)).astype(np.float32)
    PC = np.zeros((B,2)).astype(np.float32)
    RMSE = np.zeros((B,2)).astype(np.float32)
    PSNR = np.zeros((B,2)).astype(np.float32)    
    for i1 in tqdm(range(0,B)):
        f1,f2,f3,f4=FoM(Y[i1,:,:],Ynet[i1,:,:])
        SSIM[i1,0]=f1; PC[i1,0]=f2; RMSE[i1,0]=f3; PSNR[i1,0]=f4
        f1,f2,f3,f4=FoM(Y[i1,:,:],Xi[i1,:,:])
        SSIM[i1,1]=f1; PC[i1,1]=f2; RMSE[i1,1]=f3; PSNR[i1,1]=f4
    
    if config.dasorlbp == 'das':
        print('\n')
        print('############################################################### \n')
        print('Metrics results NET: \n', 'SSIM: ',round(np.mean(SSIM[:,0]),3),' +/- ',round(np.std(SSIM[:,0], ddof=1),3), ' PC: ', round(np.mean(PC[:,0]),3),' +/- ',round(np.std(PC[:,0], ddof=1),3), ' RMSE: ', round(np.mean(RMSE[:,0]),3),' +/- ',round(np.std(RMSE[:,0], ddof=1),3), ' PSNR: ', round(np.mean(PSNR[:,0]),3),' +/- ',round(np.std(PSNR[:,0], ddof=1),3))
        print('Metrics results DAS: \n', 'SSIM: ',round(np.mean(SSIM[:,1]),3), ' PC: ', round(np.mean(PC[:,1]),3), ' RMSE: ', round(np.mean(RMSE[:,1]),3), ' PSNR: ', round(np.mean(PSNR[:,1]),3))
        print('\n')
        print('############################################################### \n')
    else:
        print('\n')
        print('############################################################### \n')
        print('Metrics results NET: \n', 'SSIM: ',round(np.mean(SSIM[:,0]),3), ' PC: ', round(np.mean(PC[:,0]),3), ' RMSE: ', round(np.mean(RMSE[:,0]),3), ' PSNR: ', round(np.mean(PSNR[:,0]),3))
        print('Metrics results LBP: \n', 'SSIM: ',round(np.mean(SSIM[:,1]),3), ' PC: ', round(np.mean(PC[:,1]),3), ' RMSE: ', round(np.mean(RMSE[:,1]),3), ' PSNR: ', round(np.mean(PSNR[:,1]),3))
        print('\n')
        print('############################################################### \n')          
    
    return Xi,Y,Ynet,SSIM,PC,RMSE,PSNR,SNR

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    
    # Testing hyperparameters
    config = TestingConfig()
    
    #Xdas,Xlbp,Y,Ynet = predict()
    X,Y,Ynet,SSIM,PC,RMSE,PSNR,SNR = predict()
