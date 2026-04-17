import logging
import time
from pathlib import Path

class TrainingLogger:
    def __init__(self, log_dir="logs", log_interval=50, model_name="model"):
        self.log_interval = log_interval
        self.model_name = model_name
        self.training_start_time = None
        self.best_val_loss = float("inf")
        
        log_dir = Path(log_dir)
        log_dir.mkdir(exist_ok=True)
        
        log_file = log_dir / f"{model_name}_training_{time.strftime('%Y%m%d_%H%M%S')}.log"

        self.logger = logging.getLogger(f"training_logger_{model_name}_{time.time()}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        
        if not self.logger.handlers:

            file_handler = logging.FileHandler(log_file)
            file_handler.setFormatter(
                logging.Formatter("%(asctime)s | %(message)s")
            )

            console_handler = logging.StreamHandler()
            console_handler.setFormatter(
                logging.Formatter("%(message)s")
            )

            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
        
        self.logger.info("=" * 60)
        self.logger.info(f"Logging to {log_file}")
        self.logger.info("=" * 60)
    
    def on_training_begin(self, total_epochs):
        self.training_start_time = time.time()
        self.logger.info(f"Model: {self.model_name}")
        self.logger.info(f"Start training for {total_epochs} epochs.")
    
    def on_train_end(self):
        total_time = time.time() - self.training_start_time
        self.logger.info("=" * 60)
        self.logger.info("Training complete.")
        self.logger.info(
            f"Total training time: {total_time:.2f}s "
            f"({total_time/60:.2f} min)"
        )
        self.logger.info(f"Best validation loss: {self.best_val_loss:.4f}")
        self.logger.info("=" * 60)
        
    def on_epoch_begin(self, epoch, total_epochs):
        self.epoch_start_time = time.time()
        self.logger.info(f"\nEpoch {epoch + 1}/{total_epochs}")
        
    def on_epoch_end(self, epoch, logs):
        epoch_time = time.time() - self.epoch_start_time
        train_dice_loss = logs["train_dice_loss"]
        val_dice_loss = logs["val_dice_loss"]

        if val_dice_loss < self.best_val_loss:
            self.best_val_loss = val_dice_loss
            best_marker = " <-- new best"
        else:
            best_marker = ""
            
        self.logger.info(
            f"Epoch {epoch + 1} | "
            f"Train Dice Loss: {logs['train_dice_loss']:.4f} | "
            f"Validation Dice Loss: {logs['val_dice_loss']:.4f} | "
            f"LR: {logs['lr']:.6f} | "
            f"Time: {epoch_time:.2f}s ({epoch_time/60:.2f} min)"
            f"{best_marker}"
        )
        
    def on_batch_end(self, batch, logs=None):
        if (batch + 1) % self.log_interval == 0:
            self.logger.info(
                f"Batch {batch + 1} |" 
                f"Loss {logs['loss']:.4f} |" 
                f"Batch time: {logs['batch_time']:.4f}s")
            
    def info(self, msg):
        self.logger.info(msg)