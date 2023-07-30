import numpy as np
import os
#import gc
import cv2
from tqdm import tqdm
from dataclasses import dataclass

from OAT import createForwMatdotdet, applyDAS

# ---------------------------------------------------------------------------
"""
For convenience, create a TrainingConfig class containing the training 
hyperparameters:
"""
@dataclass
class DatasetConfig:    
    cache_dir = os.path.join(os.getcwd(), 'data') 
    datafilename = 'train_data.npy'  # Dataset for training 5652 images
    #datafilename = 'retinalbloodvessels256TRAINAUG.npy'  # Dataset for training with GAN augmentation  5652x5=28260?
    #datafilename = 'retinalbloodvessels256TEST.npy' # Dataset for testing 600 images
    dataimgsize = 256
    
    Ns = 32         # number of detectors
    Nt = 512        # number of time samples
    dx = 0.1e-3     # pixel size  in the x direction [m] 
    nx = 128        # number of pixels in the x direction for a 2-D image region
    dsa = 22.50e-3  # radius of the circunference where the detectors are placed [m]
    arco = 360      # arc of the circunferencewhere the detectors are placed
    vs = 1500       # speed of sound [m/s]
    to = 5e-6       # initial time [s]
    tf = to + 25e-6 # final time [s] 
          
    nsnr = 1        # Number of different SNR
    smax = 0.005    # Maximum noise noise standard deviation
    # Adding noise
        # nru = 1e-5 -> 90 dB (perfecto)
        # nru = 1e-4 -> 70 dB (poco ruido)
        # nru = 1e-3 -> 50 dB (ruido leve)
        # nru = 1e-2 -> 30 dB (ruidoso)  ---> esto equivale a una medición muy ruidosa
        # nru = 1e-1 -> 10 dB (muy ruidoso, casi ni se nota la señal OA)
    OneSNR = False
    
    save_sinogram = False
    augmentation = True
    
    detDAS = True
    detLBP = True
       
    date = '27jul23'
    
    shuffledata = True
    delzeroimages = True
    
    vmax = 1
    vmin = 0
    
    variablename = date  # name set for training 
    #variablename = 'a'+ date  # name set for training using GAN augmentation 
    #variablename = 't' + date  # name set for testing 
    
# ---------------------------------------------------------------------------
def augmentate_retina(MI):

    # Number of images without augmentation
    Ni = int(np.size(MI, 1))
    nx = int(np.sqrt(MI.shape[0]))
    
    ap = 3
    MI2 = np.zeros((MI.shape[0],ap*Ni))
    print(MI.shape)
    print(MI.shape[0])
    print(MI2.shape)
    cont = -1*ap
    
    for i0 in range(0,Ni):
        cont = cont + ap
        MI2[:, cont] = MI[:,i0]
        aux = np.reshape(MI[:,i0],(nx,nx))
        MI2[:,cont + 1]=np.reshape(aux[:,::-1],(int(nx**2),)) # Horizontal flip
        MI2[:,cont + 2]=np.reshape(aux[::-1,:],(int(nx**2),)) # Vertical flip
        
    return MI2

# -----------------------------------------------------------------------------
def numpynorm(x,vmax,vmin):
    y = (x-np.min(x))/(np.max(x)-np.min(x))
    y = y*(vmax-vmin) + vmin
    return y

# ---------------------------------------------------------------------------
def create_trainatestdata(): # environments: different position uncertainties

    print('Creating data for training...')
    
    config = DatasetConfig()
    
    print('Loading truth images...')
    IM = np.load(os.path.join(config.cache_dir, config.datafilename))  # (nx*nx,Ni)
    
    # Data augmentation?
    if config.augmentation:
        print('Doing basic augmentation (horizontal and vertical flip)...')
        IM = augmentate_retina(IM)
    
    # Creating Forward Model-based Matrix
    print('Creating Forward Model-based Matrix...')
    Ao = createForwMatdotdet(config.Ns,config.Nt,config.dx,config.nx,config.dsa,config.arco,config.vs,config.to,config.tf)
    #Ao = np.zeros((config.Ns*config.Nt,config.nx**2))
    Ao=Ao.astype(np.float32)
    print('done')
       
    # DATASET FOR TRAINING    
    Ni = int(IM.shape[1]) # total images

    Xdas = np.zeros((Ni*config.nsnr, config.nx, config.nx)) # corrupted pre-image
    Xlbp = np.zeros((Ni*config.nsnr, config.nx, config.nx)) # corrupted pre-image
    Y = np.zeros((Ni*config.nsnr, config.nx, config.nx)) # true image low resolution
    SNR = np.zeros(Ni*config.nsnr) # Signal to noise ratio of the corrupted sinogram
    SNRa1 = np.zeros(config.Ns) # SNR of each detectors
    
    print('Obtaining reconstructed OA images (LBP/DAS)...')
    cont = -config.nsnr # Counter index
    for i in tqdm(range(Ni)):
        cont = cont + config.nsnr
        h = IM[0:,i]
        h=h.astype(np.float32)
        if config.dataimgsize != config.nx:
            h = cv2.resize(np.reshape(h, (config.dataimgsize, config.dataimgsize)),(config.nx,config.nx),interpolation = cv2.INTER_LINEAR)
        
        h = numpynorm(h,config.vmax,config.vmin) # set data between vmin and vmax
        #h = h/np.max(h) # force maximum value = 1
        Y[cont] = h.reshape(config.nx, config.nx)  # Truth image for input without noise
        h = h.ravel()
                
        S = Ao @ h
        Sm = np.reshape(S,(config.Ns,config.Nt))
            
        rm = 0  # white noise mean value
        nru = np.random.uniform(0, config.smax, 1)[0]
        rstd = nru * np.max(np.abs(S))  # noise standard deviation 
            
        ruido = np.random.normal(rm, rstd, (config.Ns,config.Nt))
        ruido = ruido.astype(np.float32)
            
        for i2 in range(config.Ns):
            SNRa1[i2]= 20 * np.log10(np.max(np.abs(Sm[i2,:])) / np.abs(np.std(ruido[i2,:])))  # 
            
        SNR[cont] = np.mean(SNRa1) # Sinogram SNR with noise 

        if config.detDAS:
            aux = applyDAS(config.Ns,config.Nt,config.dx,config.nx,config.dsa,config.arco,config.vs,config.to,config.tf,Sm+ruido) # DAS with noise
            aux = numpynorm(aux,config.vmax,config.vmin)
            #aux = aux/np.max(np.abs(aux.ravel()))
            Xdas[cont] = aux.reshape(config.nx,config.nx)
            
        if config.detLBP:
            aux = Ao.T@((Sm + ruido).ravel()) # LBP with noise
            aux = numpynorm(aux,config.vmax,config.vmin)
            #aux = aux/np.max(np.abs(aux.ravel()))
            Xlbp[cont] = aux.reshape(config.nx,config.nx)

            
        #print(' Image: ', i + 1, ' SNR(dB): ', int(SNR[i]))
            
    if config.shuffledata:
        # Shuffle data
        print('Shuffling data...')
        indpat = np.arange(0, Y.shape[0], dtype=int)  
        ida = np.random.permutation(indpat)
        Xdas = Xdas[ida, :, :]
        Xlbp = Xlbp[ida, :, :]
        Y = Y[ida, :, :]
        SNR = SNR[ida]

    if config.delzeroimages:
        # Eliminate zeros images
        print('Eliminating zero images...')
        index=np.array([],dtype=int);
        for i4 in range(0,Y.shape[0]):
            if (not np.any(Y[i4,:,:]))==True:
                print(i4)
                index = np.append(index,i4)
        Xdas = np.delete(Xdas,index,axis=0)
        Xlbp = np.delete(Xlbp,index,axis=0)
        Y = np.delete(Y,index,axis=0)
        SNR = np.delete(SNR,index)
   
    print('Saving training data...')
    np.save(os.path.join(config.cache_dir, 'Xdas'+config.variablename), Xdas)
    np.save(os.path.join(config.cache_dir, 'Xlbp'+config.variablename), Xlbp)
    np.save(os.path.join(config.cache_dir, 'Y'+config.variablename), Y)
    np.save(os.path.join(config.cache_dir, 'SNR'+config.variablename), SNR)
   
    print('Done!')

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    
    config = DatasetConfig()
    
    create_trainatestdata()    