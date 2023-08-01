# Importing the necessary libraries:

#import os
import logging
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os

from SLcheckpoint import load_ckp, save_ckp

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split

from fdunetln import FDUNet



# ---------------------------------------------------------------------------
class ClearCache:
    # Clearing GPU Memory
    def __enter__(self):
        torch.cuda.empty_cache()

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()

# ---------------------------------------------------------------------------
def gettraindata(cache_dir, n_name, dasorlbp):
    
    print('Obtaining data for training...')

    Y = np.load(os.path.join(cache_dir, 'Y'+n_name+'.npy')) # True image
        
    Y=Y.astype(np.float32)

    if dasorlbp == 'das':
        Xdas = np.load(os.path.join(cache_dir, 'Xdas'+n_name+'.npy')) # Noisy image obtained with DAS
        Xdas=Xdas.astype(np.float32)
        return Xdas,Y
    else:
        Xlbp = np.load(os.path.join(cache_dir, 'Xlbp'+n_name+'.npy')) # Noisy image obtained with LBP
        Xlbp=Xlbp.astype(np.float32)
        return Xlbp,Y
    

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
def get_trainloader(X, Y, val_percent, batch_size): 
        
    dataset_train = OAImageDataset(X, Y)
    
    # Split into train / validation partitions
    n_val = int(len(dataset_train) * val_percent)
    n_train = len(dataset_train) - n_val
    train_set, val_set = random_split(dataset_train, [n_train, n_val], generator=torch.Generator().manual_seed(0))
    
    # Create data loaders
    #loader_args = dict(batch_size=batch_size, num_workers=8, pin_memory=True) # for local uncomment this
    loader_args = dict(batch_size=batch_size, num_workers=2, pin_memory=True) # for google_colab uncomment this
    train_loader = DataLoader(train_set, shuffle=True,  drop_last=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args) #drop_last=True, drop the last batch if the dataset size is not divisible by the batch size.
    
    return train_loader, val_loader, n_train, n_val

# ---------------------------------------------------------------------------
def train_net(config):
    
    # Use the context manager
    with ClearCache():
        # Train parameters
        img_size = config.image_size    
        batch_size = config.train_batch_size
        epochs = config.num_epochs
        lr = config.learning_rate
        ckp_last = config.ckp_last
        ckp_best = config.ckp_best
        cache_dir = config.cache_dir
        n_name = config.nname
        logfilename = config.logfilename
        continuetrain = config.continuetrain
        plotresults = config.plotresults
   
        # Set device
        device = ""
    
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        print(f"Device to be used: {device}")
    
        # Create the network    
        net = FDUNet(img_size,img_size).to(device=device)
        #print(net)
    
        # Number of net parameters
        NoP = sum(p.numel() for p in net.parameters())
        print(f"The number of parameters of the network to be trained is: {NoP}")    
    
        # Define loss function and optimizer and the the learning rate scheduler
        optimizer = Adam(net.parameters(), lr=lr, betas=(0.9, 0.999))
    
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,factor=0.5,patience=2,threshold=0.005,eps=1e-6,verbose=True)
       
        LossFn = nn.MSELoss()
        
        # Get data
        X,Y = gettraindata(cache_dir, n_name, config.dasorlbp)
        if config.dasorlbp == 'das':
            print('Training with DAS images')
        else:
            print('Training with LBP images')
        
        # Create data loader
        val_percent = config.val_percent
        X = torch.as_tensor(X).type(torch.float32) 
        Y = torch.as_tensor(Y).type(torch.float32) 
        train_loader, val_loader, n_train, n_val = get_trainloader(X, Y, val_percent, batch_size)
    
        # Initialize logging and initialize weights or continue a previous training 
    
        if continuetrain:
            net, optimizer, last_epoch, valid_loss_min = load_ckp(ckp_last, net, optimizer)
            print('Values loaded:')
            #print("model = ", net)
            print("optimizer = ", optimizer)
            print("last_epoch = ", last_epoch)
            print("valid_loss_min = ", valid_loss_min)
            print("valid_loss_min = {:.6f}".format(valid_loss_min))
            start_epoch = last_epoch + 1
            lr = optimizer.param_groups[0]['lr']
            logging.basicConfig(filename=logfilename,format='%(asctime)s - %(message)s', level=logging.INFO)
            logging.info(f'''Continuing training:
                Epochs:                {epochs}
                Batch size:            {batch_size}
                Initial learning rate: {lr}
                Training size:         {n_train}
                Validation size:       {n_val}
                Device:                {device.type}
                ''')
        else:
            # Apply the weights_init function to randomly initialize all weights
            net.apply(initialize_weights)
            start_epoch = 1
            valid_loss_min = 100
            logging.basicConfig(filename=logfilename, filemode='w',format='%(asctime)s - %(message)s', level=logging.INFO)
            logging.info(f'''Starting training:
                Epochs:                {epochs}
                Batch size:            {batch_size}
                Initial learning rate: {lr}
                Training size:         {n_train}
                Validation size:       {n_val}
                Device:                {device.type}
                ''')
    
    
                # Print model
                # print(net)

        # Begin training
        TLV=np.zeros((epochs,)) #vector to record the train loss per epoch 
        VLV=np.zeros((epochs,)) #vector to record the validation loss per epoch
        EV=np.zeros((epochs,)) # epoch vector to plot later
        global_step = 0
    
        #for epoch in range(epochs):
        for epoch in range(start_epoch, start_epoch+epochs):
            net.train() # Let pytorch know that we are in train-mode
            epoch_loss = 0.0
            epoch_val_loss = 0.0
            with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs+start_epoch-1}', unit='sino') as pbar:
                for x,y in train_loader:
                    # clear the gradients
                    optimizer.zero_grad(set_to_none=True)
                    # input to device
                    x = x.to(device=device)
                    x = torch.unsqueeze(x,1)
                    x = x.type(torch.float32)
                    # net prediction
                    pred = net.forward(x)
                    pred = torch.squeeze(pred,1)
                    # True image
                    y = y.to(device=device) # (-1,nx,nx)
                    y = y.type(torch.float32)
                    # calculate image loss
                    train_loss = LossFn(pred,y)
                    # credit assignment
                    train_loss.backward()
                    # update model weights
                    optimizer.step()
                
                    pbar.update(x.shape[0])
                    global_step += 1
                    
                    epoch_loss += train_loss.item()
                
                    pbar.set_postfix(**{'loss (batch)': train_loss.item()})
            
                epoch_train_loss = epoch_loss / len(train_loader)

            # Evaluation round
            with torch.no_grad():
                for xv,yv in tqdm(val_loader, total=len(val_loader), desc='Validation round', position=0, leave=True):
                    # input and truth to device
                    xv = xv.to(device=device)
                    xv = torch.unsqueeze(xv,1)
                    xv = xv.type(torch.float32)
                    # net prediction
                    predv = net.forward(xv)
                    predv = torch.squeeze(predv,1)
                    # True image
                    yv = yv.to(device=device) # (-1,nx,nx)
                    yv = yv.type(torch.float32)
                    # calculate loss
                    val_loss = LossFn(predv,yv)
                    epoch_val_loss += val_loss.item()
                    
            epoch_val_loss = epoch_val_loss / len(val_loader)
                    
            # Scheduler ReduceLROnPlateau
            scheduler.step(epoch_val_loss)
        
            # logging validation score per epoch
            logging.info(f'''Epoch: {epoch} - Validation score: {np.round(epoch_val_loss,5)}''')
        
            # print training/validation statistics 
            #print('\n Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
            #    epoch,
            print('\n Training Loss: {:.5f} \tValidation Loss: {:.5f}'.format(
                    epoch_train_loss,
                    epoch_val_loss
                    ))
        
            # Loss vectors for plotting results
            TLV[epoch-start_epoch]=epoch_train_loss
            VLV[epoch-start_epoch]=epoch_val_loss
            EV[epoch-start_epoch]=epoch
        
            # create checkpoint variable and add important data
            checkpoint = {
                    'epoch': epoch,
                    'valid_loss_min': epoch_val_loss,
                    'state_dict': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    }
        
            # save checkpoint
            save_ckp(checkpoint, False, ckp_last, ckp_best)
        
        
            # save the model if validation loss has decreased
            if epoch_val_loss <= valid_loss_min:
                print('\n Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(valid_loss_min,epoch_val_loss),'\n')
                # save checkpoint as best model
                save_ckp(checkpoint, True, ckp_last, ckp_best)
                valid_loss_min = epoch_val_loss
                logging.info(f'Val loss deccreased on epoch {epoch}!')
        

    
        if plotresults:
            plt.figure();
            plt.grid(True,linestyle='--')
            plt.xlabel('epoch'); plt.ylabel('Loss')
            plt.plot(EV,TLV,'--',label='Train Loss')
            plt.plot(EV,VLV,'-',label='Val Loss')
            plt.legend(loc='best',shadow=True, fontsize='x-large')
    
        return EV,TLV,VLV
               
# --------------------------------------------------------------------------- 
def initialize_weights(m):
    if isinstance(m,(nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.normal_(m.weight.data,0.0,0.02)
        if m.bias is not None:
            nn.init.constant_(m.bias.data,0)
    elif isinstance(m,(nn.BatchNorm2d,nn.LayerNorm)):
        nn.init.normal_(m.weight.data,1.0,0.02)
        nn.init.constant_(m.bias.data,0)

# ---------------------------------------------------------------------------
#if __name__=='__main__':
#    test()
