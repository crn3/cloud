import segmentation_models_pytorch as smp

from datasets.prepare_dataset import get_dataloaders
from utils.early_stopping import EarlyStopping

import torch
import torch.backends.cudnn as cudnn
from torch.amp import autocast, GradScaler
from pathlib import Path
import time
from utils.training_logger import TrainingLogger
from pytorch_toolbelt.losses import DiceLoss
import matplotlib.pyplot as plt

def main():

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    
    cudnn.benchmark = True

    print("CUDA available:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("CUDA device:", torch.cuda.get_device_name(0))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)

    EPOCHS = 50
    
    #########################################
    ###### Choose which model to train ######
    #########################################
    
    MODEL_TYPE = "unetplusplus" 
    # MODEL_TYPE = "unet"
    
    if MODEL_TYPE == "unet":
        model = smp.Unet(
            encoder_name="mobilenet_v2",
            encoder_weights=None, # pretrained weights expext 3 channels?
            in_channels = 4,
            classes=2
            )
    elif MODEL_TYPE == "unetplusplus":
        model = smp.UnetPlusPlus(
            encoder_name="mobilenet_v2",
            encoder_weights=None, 
            in_channels = 4,
            classes=2
            )
    else:
        raise ValueError("Invalid model type.")
    
    ########################################
    ### Create folder for trained models ###
    ########################################
    
    model_name = model.__class__.__name__
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    
    model_dir = Path("models") / f"{model_name}_{timestamp}"
    model_dir.mkdir(parents=True, exist_ok=True)
    
    MODEL_SAVE_PATH = model_dir / f"{model_name}.pth"
    
    model.to(device)
    
    train_dl, valid_dl = get_dataloaders(batch_size=16, num_workers=4)
    
    ################################
    ### Dice score and dice loss ### 
    ################################
    
    # Dice = 2 |A∩B| / (|A|+|B|) = 2 TP / (2 TP + FP + FN)
    def dice_score(pred, target, eps=1e-6):
        pred = torch.argmax(pred, dim=1)
        intersection = (pred == target).sum().float()
        return (2. * intersection + eps) / (pred.numel() + target.numel() + eps)

    loss_fn = DiceLoss(
        mode="multiclass",
        from_logits=True,
        smooth=1.0,
        ignore_index=None
    )
    
    #################
    ### Optimiser ###
    #################
        
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    #################################
    ### Automatic mixed precision ###
    #################################
    
    use_amp = device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)
    
    ###############################
    ### Learning rate scheduler ###
    ###############################
    
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimiser,
    mode='min',
    factor=0.5,
    patience=2,
    threshold=1e-3
)
    
    #####################
    ### Early stopper ###
    #####################
    
    early_stopper = EarlyStopping(patience=5, min_delta=1e-3) # try other delta values?
    
    ##############
    ### Logger ###
    ##############
    
    logger = TrainingLogger(log_interval=50, model_name=model_name)
    logger.on_training_begin(EPOCHS)
    
    ################################################
    ### Store training metrics for visualisation ###
    ################################################
        
    train_losses = []
    val_losses = []
    train_scores = []
    val_scores = []
    lrs = []
    
    ################
    ### Training ###
    ################
    
    best_val_loss = float("inf")
    
    for epoch in range(EPOCHS):
    
        logger.on_epoch_begin(epoch, EPOCHS)
        epoch_start = time.time()
        
        model.train()
        train_dice_loss = 0
        train_dice_score = 0
        
        for batch_idx, (xb, yb) in enumerate(train_dl):
            batch_start = time.time()
            
            xb = xb.to(device)
            yb = yb.to(device)
            
            optimiser.zero_grad()
            
            with autocast(device_type=device.type, enabled=use_amp):
                pred = model(xb)
                loss = loss_fn(pred, yb)
            
            ##################################
            ### Adjusting model using loss ###
            ##################################
            
            scaler.scale(loss).backward()
            scaler.step(optimiser)
            scaler.update()
            
            with torch.no_grad():
                dice = dice_score(pred, yb)
            
            batch_time = time.time() - batch_start
            
            logger.on_batch_end(
                batch_idx,
                {
                    "loss": loss.item(),
                    "batch_time": batch_time ## add more here???
                    }
                )
            
            train_dice_loss += loss.item()
            train_dice_score += dice.item()
        
        train_dice_loss /= len(train_dl)
        train_dice_score /= len(train_dl)
        
        ##################
        ### Validation ###
        ##################
        
        model.eval()
        val_dice_loss = 0
        val_dice_score = 0
        
        with torch.no_grad():
            for xb, yb in valid_dl:
                xb = xb.to(device)
                yb = yb.to(device)

                with autocast(device_type=device.type, enabled=use_amp):
                    pred = model(xb)
                    loss = loss_fn(pred, yb)
                    
                dice = dice_score(pred, yb)

                val_dice_loss += loss.item()
                val_dice_score += dice.item()

        val_dice_loss /= len(valid_dl)
        val_dice_score /= len(valid_dl)
        
        ######################
        ### Scheduler step ###
        ######################
        
        scheduler.step(val_dice_loss)
        
        current_lr = optimiser.param_groups[0]['lr']
        
        ###############
        ### Logging ###
        ###############

        print(f"Train dice loss: {train_dice_loss:.4f}")
        print(f"Train dice score: {train_dice_score:.4f}")
        print(f"Val dice loss: {val_dice_loss:.4f}")
        print(f"Val dice score: {val_dice_score:.4f}")
        print(f"Learning rate: {current_lr:.6f}")
        
        train_losses.append(train_dice_loss)
        val_losses.append(val_dice_loss)
        train_scores.append(train_dice_score)
        val_scores.append(val_dice_score)
        lrs.append(current_lr)
        
        logger.on_epoch_end(
            epoch,
            {
                "train_dice_loss": train_dice_loss,
                "train_dice_score": train_dice_score,
                "val_dice_loss": val_dice_loss,
                "val_dice_score": val_dice_score,
                "lr" : current_lr
                }
            )
        
        epoch_time = time.time() - epoch_start
        print(f"Epoch time: {epoch_time:.2f} seconds ({epoch_time/60:.2f} minutes)")    
        
        #######################
        ### Save best model ###
        #######################
        
        if val_dice_loss < best_val_loss:
            best_val_loss = val_dice_loss
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimiser_state_dict': optimiser.state_dict(),
                'val_loss': val_dice_loss
            }, MODEL_SAVE_PATH)
            
            print("Model saved.")
        
        ######################
        ### Early stopping ###
        ######################
        
        if early_stopper.early_stop(val_dice_loss):
            logger.info("Early stop triggered.")
            break
        
    
    ############################
    ### end of training loop ###
    ############################      
    
    logger.on_train_end()
    print("\nTraining complete!")
    
    ################
    ### Plotting ###
    ################
    
    epochs = range(1, len(train_losses) + 1)
    
    plt.figure(figsize=(12, 5))
    
    ### Dice loss ###
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='train')
    plt.plot(epochs, val_losses, label='val')
    plt.title('Dice Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    ### Dice score ###
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_scores, label='train')
    plt.plot(epochs, val_scores, label='val')
    plt.title('Dice Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(model_dir / "training_curves.png", dpi=300)
    plt.close()
    
    
    plt.figure()
    plt.plot(range(1, len(lrs)+1), lrs)
    plt.title("Learning Rate")
    plt.xlabel("Epoch")
    plt.ylabel("Learning rate")
    plt.savefig(model_dir / "learning_rate.png", dpi=300)
    plt.close()
    
if __name__ == "__main__":
    main()