import json
import shutil
import torch
import os
import time
import torch
from datetime import  timedelta

from functools import *
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from fastai.vision.all import *
from torchsampler import ImbalancedDatasetSampler


from os import listdir, makedirs, path
from vtcv_fine_grained_classification.src.configs.train_config import TrainConfig
from vtcv_fine_grained_classification.src.configs.config import ExperimentationConfig
from vtcv_fine_grained_classification.src.configs.paths_config import PathsConfig
from vtcv_fine_grained_classification.src.models.region import *
from vtcv_fine_grained_classification.src.models.model import MainModel
from vtcv_fine_grained_classification.src.utils.utils import save_time_tracking, log_information
from vtcv_fine_grained_classification.src.models.build_model import build_model, define_losses
from vtcv_fine_grained_classification.train.train_utils import load_nets_and_freeze_param




def make_dirs(dirs):
    for dir_ in dirs:
        makedirs(dir_, exist_ok=True)

class Trainer(torch.nn.Module):
    def __init__(self, config, exp_dir, model,  nets):
        super().__init__()
        self.exp_dir = exp_dir
        self.config = config
        self.device = config.device
        self.model = model
        self.nets = nets
      
        # logger.log(f"Saving experiment to {exp_dir}")        
        if not exp_dir.exists():
            make_dirs([exp_dir])

        datasheet_dest_path = exp_dir / "full_data.csv"
        # logger.log(f"Copying datasheet from {config.datasheet_path} to {datasheet_dest_path}")
        shutil.copy(config.datasheet_path, datasheet_dest_path)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


    def _save_checkpoint(self):    
        torch.save(self.model.state_dict(), self.exp_dir / f"model_last.pth")    
        

    def _load_checkpoint(self):
        model = torch.load(self.exp_dir / f"model_last.pth")
        return model

    def _reset_grad(self):    
        self.optim.zero_grad() 

    def train_(self, config, train_dataloader, valid_dataloader, device):
        best_val_loss = float('inf')
        best_val_acc = 0


        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optim, mode='min', 
                                                 factor=0.1, 
                                                 patience=config.early_stopping_kwargs["patience"], 
                                                 min_lr=config.early_stopping_kwargs["min_delta"])

        for epoch in range(1, config.epochs+1):
            start_time = time.time()
            losses_train, losses_train_crop,losses_train_drop, loss_weight_train, train_accuracy, train_accuracy_crop, \
                train_accuracy_drop, train_accuracy_image_combined = self.train_epoch(config, self.model, train_dataloader, \
                                                                                      device, cur_epoch=epoch)   
             
            losses_val, losses_val_crop, losses_val_drop, loss_weight_val, val_accuracy, valid_accurcy_crop, \
                valid_accurcy_drop, val_accuracy_image_combined = self.train_epoch(config, self.model, valid_dataloader, \
                                                                                    device, valid=True) 
            
         
            elapsed = time.time() - start_time
            elapsed = str(timedelta(seconds=elapsed))[:-7]

            # Log information
            log_information(config, elapsed, (losses_train, losses_val), \
                            (losses_train_crop, losses_val_crop), (losses_train_drop, losses_val_drop), \
                            (loss_weight_train, loss_weight_val), (train_accuracy, val_accuracy), \
                            (train_accuracy_crop, valid_accurcy_crop), (train_accuracy_drop, valid_accurcy_drop), \
                            (train_accuracy_image_combined, val_accuracy_image_combined), self.optim.param_groups[0]['lr'], \
                            epoch)
            


            # log to tensorboard
            # train
            writer.add_scalar('Loss/train_criterion_loss', losses_train['criterion_loss'], epoch)
            writer.add_scalar('Accuracy/train_accuracy_image', train_accuracy['accuracy_image'], epoch)
      
            # val
            writer.add_scalar('Loss/val_criterion_loss', losses_val['criterion_loss'], epoch)           
            writer.add_scalar('Accuracy/val_accuracy_image', val_accuracy['accuracy_image'], epoch)
            
            writer.flush()

            if  best_val_loss > losses_val['criterion_loss'] or best_val_acc < val_accuracy_image_combined['accuracy_image_combined']:
                print(f"Saving model at epoch **** {epoch} *****")
                best_val_loss = losses_val['criterion_loss']
                best_val_acc = val_accuracy_image_combined['accuracy_image_combined']
                torch.save(self.model.state_dict(), self.exp_dir / f"model_best.pth")
                torch.save(self.nets.base_model.state_dict(), self.exp_dir / f"model_best_base.pth")
                torch.save(self.nets.emb_model.state_dict(), self.exp_dir / f"model_best_tax.pth")
                
           
            scheduler.step(losses_val['criterion_loss'])
        writer.close()
       


    def train_epoch(self, config, model, dataloader, device, valid=False, cur_epoch=0,):           
        
        total_criterion_loss = 0
        total_criterion_loss_crop = 0
        total_criterion_loss_drop = 0
        total_weighted_loss = 0


        correct_image = 0     
        correct_image_crop = 0
        correct_image_drop = 0
        correct_image_combined = 0

        if config.track_time:
            durations_over_all_batches = []


        for cur_batch, (images, labels) in enumerate(tqdm(dataloader)):
            images, labels = images.to(device), labels.to(device)

            # Evaluate and compute losses          
            if valid:
                model.eval()
                criterion_loss, outputs = model(images, labels, mode='Valid', cur_epoch=cur_epoch, cur_batch=cur_batch)
            else:
                model.train()
                criterion_loss, outputs = model(images, labels, mode='Train', cur_epoch=cur_epoch)   
            
            loss_diversity =  outputs['loss_diversity']

            _ , crop_loss, drop_loss, combined_loss = criterion_loss
            weighted_loss = combined_loss * config.weighted_loss_parameters["combined_loss_weight"] \
                            + loss_diversity * config.weighted_loss_parameters["diversity_loss_weight"]

           
            # Perform backpropagation for the combined weighted loss
            if not valid:
                if config.track_time:
                    start_time = time.time()

                self.optim.zero_grad()  # Clear gradients before backward
                weighted_loss.backward()  # Single backward pass
                self.optim.step()  # Update optimizer

                if config.track_time:
                    duration = outputs["duration"]
                    duration["Duration for gradient calculation"] = time.time() - start_time
                    durations_over_all_batches.append(duration)

            total_weighted_loss += weighted_loss.sum().item()
            total_criterion_loss += combined_loss.sum().item()
            total_criterion_loss_crop += crop_loss.sum().item()
            total_criterion_loss_drop += drop_loss.sum().item()


            # Calculate accuracies
            _, predicted = torch.max(outputs['logits'], 1)
            correct_image += (predicted == labels).sum().item()
        
            if config.weight_logits_crop != 0 and outputs['logits_crop'] is not None:
                _, predicted_crop = torch.max(outputs['logits_crop'], 1)
                correct_image_crop += (predicted_crop == labels).sum().item()

            if config.weight_logits_drop != 0 and outputs['logits_drop'] is not None: 
                _, predicted_drop = torch.max(outputs['logits_drop'], 1)  
                correct_image_drop += (predicted_drop == labels).sum().item()
            
            _, predicted_combined = torch.max(outputs['combined_logits'], 1)  
            correct_image_combined += (predicted_combined == labels).sum().item()
            

        # Calculate average losses and accuracies
        avg_loss_criterion = total_criterion_loss / len(dataloader.dataset)
        avg_loss_criterion_crop = total_criterion_loss_crop / len(dataloader.dataset)
        avg_loss_criterion_drop = total_criterion_loss_drop / len(dataloader.dataset)
        avg_weighted_loss = total_weighted_loss / len(dataloader.dataset)
        
        accuracy_image = 100 * correct_image / len(dataloader.dataset)
        accuracy_image_crop = 100 * correct_image_crop / len(dataloader.dataset)
        accuracy_image_drop = 100 * correct_image_drop / len(dataloader.dataset)
        accuracy_image_combined = 100 * correct_image_combined / len(dataloader.dataset)


        # Calculate and save average time durations for different training steps in a json file
        #if config.track_time and not valid and cur_epoch % config.save_every == 1:
        if config.track_time and not valid and cur_epoch % 2 == 1:
            exp_dir = Path(config.save_dir) / config.exp_name
            filepath = exp_dir / "time_track.json"
            save_time_tracking(durations_over_all_batches, filepath, cur_epoch)
        
        return {'criterion_loss': avg_loss_criterion }, \
            {'criterion_loss_crop': avg_loss_criterion_crop},\
            {'criterion_loss_drop': avg_loss_criterion_drop},\
            {'weighted_loss': avg_weighted_loss},\
            {'accuracy_image': accuracy_image},\
            {'accuracy_image_crop': accuracy_image_crop},\
            {'accuracy_image_drop': accuracy_image_drop}, \
            {'accuracy_image_combined': accuracy_image_combined}





def train(config: ExperimentationConfig):
    """Train the model with support for multi-GPU parallelism."""
    
    # Set up configuration paths
    config_paths = PathsConfig(expconfig=config)
    train_config = TrainConfig(expconfig=config, paths_config=config_paths)
    assert config.mode == "train", "Incorrect settings"
    global writer
    writer = SummaryWriter(log_dir=f'runs/{config.exp_name}')
    exp_dir = Path(config.save_dir) / config.exp_name    
    os.makedirs(exp_dir, exist_ok=True)

    # Save configurations
    with (exp_dir / "TrainConfig.json").open("w") as frozen_settings_file:
        json.dump(config.dict(exclude_none=True), frozen_settings_file, indent=2)


    # Save processed datasets
    train_config.save_processed_datasets()
    
    
    # Define network modules
    num_classes = train_config.get_num_classes(config.datasheet_path)     
    device = config.device
    config.num_classes = num_classes
    nets = build_model(config, device,  num_classes=num_classes)



    # Loading initial networks and freeze taxonomy network parameters
    nets = load_nets_and_freeze_param(config, nets)
    


    # Create model
    criterion = train_config.get_loss(num_classes, device)    
    losses = define_losses(criterion)
    model = MainModel(nets, losses, config)
    model = model.to(config.device)
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    
    # Define trainer
    trainer = Trainer(config, exp_dir, model, nets)


    # Load pretrained model parameters if load_pretrained_models is true
    if config.initial_model_kwargs["load_pretrained_models"]:
        model.load_state_dict(torch.load(config.initial_model_kwargs["initial_model_path"]))
    
    
    # Training and validation datasets
    train_set = train_config.get_dataset(mode="Train")
    valid_set = train_config.get_dataset(mode="Valid")
    train_dataloader = DataLoader(train_set, sampler=ImbalancedDatasetSampler(train_set), \
                                  batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, \
                                  pin_memory=True)
    valid_dataloader = DataLoader(valid_set, batch_size=config.batch_size, shuffle=True, \
                                  num_workers=config.num_workers, pin_memory=True)
    
    
    # Train the classifier
    trainer.train_(config, train_dataloader, valid_dataloader, device)



