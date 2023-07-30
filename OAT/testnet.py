from dataclasses import dataclass

import numpy as np
import os

from fdunetln import FDUNet
import torch
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset

from quality import FoM
from tqdm import tqdm

# ---------------------------------------------------------------------------
@dataclass
class TestingConfig:
    
    image_size = 128  # the image size and shape
    test_batch_size = 16
    device = 'cpu'

    cache_dir = 'data/' 
    nname = 't1' # name set for testing
    
    traindate = '24jul23' 
    
    ckp_last='fdunet' + traindate + '.pth' # name of the file of the saved weights of the trained net
    ckp_best='fdunet_best' + traindate + '.pth'
    
    logfilename = 'TrainingLog_FDUNet' + traindate + '.log'
    
    dasorlbp = 'das' #lbp

# ---------------------------------------------------------------------------
def gettestdata(cache_dir, n_name):
    
    print('Obtaining data for testing...')
    
    Xdas = np.load(os.path.join(cache_dir, 'Xdas'+n_name+'.npy')) # Noisy image obtained with DAS
    Xlbp = np.load(os.path.join(cache_dir, 'Xlbp'+n_name+'.npy')) # Noisy image obtained with LBP

    Y = np.load(os.path.join(cache_dir, 'Y'+n_name+'.npy')) # True image
    SNR = np.load(os.path.join(cache_dir, 'SNR'+n_name+'.npy')) # True image
        
    Xdas=Xdas.astype(np.float32)
    Xlbp=Xlbp.astype(np.float32)
    Y=Y.astype(np.float32)
    SNR=SNR.astype(np.float32)

    print('done')
    
    return Xdas,Xlbp,Y,SNR

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
def predict():
    
    config = TestingConfig()
    
    # Create net object
    net = FDUNet(config.image_size,config.image_size).to(device=config.device)
    # Loading best checkpoint
    checkpoint = torch.load(config.ckp_best,map_location=torch.device(config.device))
    net.load_state_dict(checkpoint['state_dict'])
    
    # Load dataset
    Xdas,Xlbp,Y,SNR = gettestdata(config.cache_dir, config.nname)
    if config.dasorlbp == 'das':
        X = Xdas
    else:
        X = Xlbp
       
    # Create data loader
    X = torch.as_tensor(X).type(torch.float32) 
    
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
    
    # Measuring the quality of the reconstruction 
    print('Calculating metrics...')
    SSIM = np.zeros((B,3)).astype(np.float32)
    PC = np.zeros((B,3)).astype(np.float32)
    RMSE = np.zeros((B,3)).astype(np.float32)
    PSNR = np.zeros((B,3)).astype(np.float32)    
    for i1 in tqdm(range(0,B)):
        f1,f2,f3,f4=FoM(Y[i1,:,:],Ynet[i1,:,:])
        SSIM[i1,0]=f1; PC[i1,0]=f2; RMSE[i1,0]=f3; PSNR[i1,0]=f4
        f1,f2,f3,f4=FoM(Y[i1,:,:],Xlbp[i1,:,:])
        SSIM[i1,1]=f1; PC[i1,1]=f2; RMSE[i1,1]=f3; PSNR[i1,1]=f4
        f1,f2,f3,f4=FoM(Y[i1,:,:],Xdas[i1,:,:])
        SSIM[i1,2]=f1; PC[i1,2]=f2; RMSE[i1,2]=f3; PSNR[i1,2]=f4
        
    print('\n')
    print('############################################################### \n')
    print('Metrics results NET: \n', 'SSIM: ',round(np.mean(SSIM[:,0]),3), ' PC: ', round(np.mean(PC[:,0]),3), ' RMSE: ', round(np.mean(RMSE[:,0]),3), ' PSNR: ', round(np.mean(PSNR[:,0]),3))
    print('Metrics results LBP: \n', 'SSIM: ',round(np.mean(SSIM[:,1]),3), ' PC: ', round(np.mean(PC[:,1]),3), ' RMSE: ', round(np.mean(RMSE[:,1]),3), ' PSNR: ', round(np.mean(PSNR[:,1]),3))
    print('Metrics results DAS: \n', 'SSIM: ',round(np.mean(SSIM[:,2]),3), ' PC: ', round(np.mean(PC[:,2]),3), ' RMSE: ', round(np.mean(RMSE[:,2]),3), ' PSNR: ', round(np.mean(PSNR[:,2]),3))
    print('\n')
    print('############################################################### \n')
    
    return Xdas,Xlbp,Y,Ynet,SSIM,PC,RMSE,PSNR,SNR

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    
    # Testing hyperparameters
    config = TestingConfig()
    
    #Xdas,Xlbp,Y,Ynet = predict()
    Xdas,Xlbp,Y,Ynet,SSIM,PC,RMSE,PSNR,SNR = predict()