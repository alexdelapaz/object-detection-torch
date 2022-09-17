import pickle 
import torch
import utils
from tqdm import tqdm
import numpy as np
import os
import copy
import sys

from detection import models

# adding torchvision vision tools folder to import modules
notebook_folders = ['vision/references/detection']
for folder in notebook_folders:
    sys.path.append(folder)

#sys.path.insert(0, '/vision')

from engine import evaluate


def fasterrcnn(model, model_path, data_loaders, epoch_count, freq_eval_save, lr=0.0001, opt='sgd'):
    
    # Training info capture
    losses_train = []
    losses_val = []
    losses_val_NEW = []
    eval_per_n_epoch = []
    best_model_weights = model.state_dict()
    current_epoch = 0

    # Loss monitoring initialization
    lowest_train_loss = 99999999
    lowest_val_loss = 99999999

    # Prior training weights and information
    if os.path.exists(model_path):
        # Stored model information
        model_info = torch.load(model_path)

        # Model weights
        best_model_weights = model_info['weights']
        model.load_state_dict(best_model_weights)

        # Prior training run epoch count
        current_epoch = model_info['epochs_trained']

        # Prior training run losses
        losses_train = model_info['losses_train']
        losses_val = model_info['losses_val']
        losses_val_NEW = model_info['losses_val']

        # Prior lowest loss for training and validation
        lowest_train_loss = min(losses_train)
        lowest_val_loss = min(losses_val)

        print(f'\nEpochs trained:\t\t{current_epoch}')
    
    # Define path to save out models over time
    weights_at_eval = '/'.join(model_path.split('/')[:-1]) + '/weights_at_eval/'

    if not os.path.exists(weights_at_eval):
        os.mkdir(weights_at_eval)

    # Assign dataloaders to variables from tuple argument
    train_loader, val_loader = data_loaders

    # Utilize GPU (if available) CPU otherwise
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Collect Model parameters to pass to optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    # Set optimizer
    if opt.lower() == 'adam':
        optimizer = torch.optim.Adam(params, lr=lr)
    elif opt.lower() == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr)

    for epoch in range(current_epoch, epoch_count):
        
        print(f'\nTraining [epoch {epoch + 1}]:\tout of {epoch_count}')
        
        ### Training ###
        model.train()
        epoch_train_losses = []
        # Process all data in the data loader 
        for imgs, annotations in tqdm(train_loader, desc = 'Training'):
            
            # Prepare images and annotations
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            
            # Calculate loss 
            loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())
            epoch_train_losses.append(losses.cpu().detach().numpy())

            # Backprop
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()

        # Train epoch done
        epoch_train_loss = np.mean(epoch_train_losses)
        losses_train.append(epoch_train_loss)
        

        ### Validation ###
        epoch_val_losses = []
        # Process all data in the data loader 
        for imgs, annotations in tqdm(val_loader, desc = 'Validation'):
            
            # Prepare images and annotations
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            
            # Calculate loss
            with torch.no_grad():
                loss_dict = model(imgs, annotations)
            losses = sum(loss for loss in loss_dict.values())
            epoch_val_losses.append(losses.cpu().detach().numpy())

        # Val epoch done
        epoch_val_loss = np.mean(epoch_val_losses)
        losses_val.append(epoch_val_loss)


        # Implementation of FasterRCNN in eval mode to calculate loss for validation
        
        with torch.no_grad():
            ### Validation ###
            epoch_val_loss = models.frcnn_evaluate_loss(model, val_loader, device)
            losses_val_NEW.append(epoch_val_loss)
        


        # Eval per freq_eval_save
        if epoch % freq_eval_save == (freq_eval_save-1):
            # update the learning rate
            # TODO lr_scheduler methods
            #lr_scheduler.step()
            
            # evaluate on the test dataset
            eval_data = evaluate(model, val_loader, device=device)
            eval_per_n_epoch.append(eval_data)

        # Display lowest training loss
        if epoch_train_loss < lowest_train_loss:
            lowest_train_loss = epoch_train_loss
        print(f'Lowest train loss: {lowest_train_loss}')
        
        # Display lowest validation loss
        # Save best weights if validation is lowest validation so far
        if epoch_val_loss < lowest_val_loss:
            best_model_weights = copy.deepcopy(model.state_dict())
            lowest_val_loss = epoch_val_loss
        print(f'Lowest val loss: {lowest_val_loss}')

        # Model information
        model_info = {'weights': best_model_weights,
                      'epochs_trained': epoch + 1,
                      'least_train_loss': lowest_train_loss,
                      'least_val_loss': lowest_val_loss,
                      'losses_train': losses_train,
                      'losses_val': losses_val,
                      'losses_val_NEW': losses_val_NEW,
                      'evals': eval_per_n_epoch}

        # Save the model information and best weights so far (from above val save) for each eval conducted
        if epoch > 0:
            if epoch % freq_eval_save == (freq_eval_save-1):
                model_path_eval = weights_at_eval + f'{epoch+1}_epochs'
                torch.save(model_info, model_path_eval)
                
                print('Saving eval run experiment artifacts at: {}'.format(model_path_eval), sep='\n')

        #  Save artifacts and model
        torch.save(model_info, model_path)
        torch.save(model, model_path+'.pt')
    
    return losses_train