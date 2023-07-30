from dataclasses import dataclass

from train import train_net

# ---------------------------------------------------------------------------
@dataclass
class TrainingConfig:
    
    image_size = 128  # the image size and shape
    
    train_batch_size = 15
    num_epochs = 50
    learning_rate = 5e-4
    val_percent = 0.2 #0.2
    
    cache_dir = 'data/' 
    nname = '1' # name set for training
    #nname = 'a1' # name set for training using GAN augmentation
    
    continuetrain = False
    plotresults = True
    traindate = '24jul23' 
    
    ckp_last='fdunet' + traindate + '.pth' # name of the file of the saved weights of the trained net
    ckp_best='fdunet_best' + traindate + '.pth'
    
    logfilename = 'TrainingLog_FDUNet' + traindate + '.log' 
    
    dasorlbp = 'das' #lbp

# ---------------------------------------------------------------------------
if __name__ == '__main__':
    
    # Training hyperparameters
    config = TrainingConfig()
    
    EV,TLV,VLV = train_net(config)    