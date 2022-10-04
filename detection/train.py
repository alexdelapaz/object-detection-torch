import pickle 
import torch
from tqdm import tqdm
import numpy as np
import os
import copy
import sys


# add torchvision vision tools folder to import modules
# path 1 is for github pull of repo
# path 2 is for ipynb use in colab
notebook_folders = ['vision/references/detection', '/content/drive/My Drive/Colab Notebooks/']
for folder in notebook_folders:
    sys.path.append(folder)
#sys.path.insert(0, '/vision')

# Import torchvision vision folder scripts from vision/references/detection
from engine import evaluate
import utils

# import light training framework for torch
from detection import models


def fasterrcnn(model, train_loader, val_loader, test_loader=None, opt=None, lr=None, epoch_count=None, freq_eval_save=None, save_path=None):
    '''
    # Tracking training loop settings and artifacts
    log_settings = ['epochs_trained', 'weights', 'losses_train']

    if val_loader is not None:
        log_settings.append('losses_val')
    if test_loader is not None:
        log_settings.append('losses_test')  
  
    log_training = {}
    for setting in log_settings:
        log_training[setting] = []
    '''


    current_epoch = 0
    best_model_weights = model.state_dict()
    losses_train = []
    losses_val = []
    eval_per_n_epoch = []



    # Prior training weights and information
    if os.path.exists(save_path):
        # Stored model information
        model_info = torch.load(save_path)

        # Model weights
        best_model_weights = model_info['weights']
        model.load_state_dict(best_model_weights)

        # Prior training run epoch count
        current_epoch = model_info['epochs_trained']

        # Prior training run losses
        losses_train = model_info['losses_train']
        losses_val = model_info['losses_val']

        # Prior lowest loss for training and validation
        lowest_train_loss = min(losses_train)
        lowest_val_loss = min(losses_val)

        print(f'\nEpochs trained:\t\t{current_epoch}')
    
    # Define path to save out models over time
    state_per_epoch = '/'.join(save_path.split('/')[:-1]) + '/state_per_epoch/'

    if not os.path.exists(state_per_epoch):
        os.mkdir(state_per_epoch)


    # Utilize GPU (if available) CPU otherwise
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.to(device)

    # Collect Model parameters to pass to optimizer
    params = [p for p in model.parameters() if p.requires_grad]

    # Set optimizer
    if opt == None:
        optimizer = torch.optim.Adam(params, lr=lr)
    elif opt.lower() == 'sgd':
        optimizer = torch.optim.SGD(params, lr=lr)

    for epoch in range(current_epoch, epoch_count):
        
        print(f'\nTraining [epoch {epoch + 1}]:\tout of {epoch_count}')
        
        # Perform foward pass, backpropagation, zero gradients
        # Calculate training loss for one eppoch, add training loss to total training loss and store in log
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
        

        # Perform foward pass, backpropagation, zero gradients
        # Calculate training loss for one eppoch, add training loss to total training loss and store in log
        # model.eval() will not work as usually intended for torchvision FRCNN due to inference outputs
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

        '''
        # Implementation of FasterRCNN in eval mode to calculate loss for validation
        # Perform foward pass, backpropagation, zero gradients
        # Calculate training loss for one eppoch, add training loss to total training loss and store in log
        with torch.no_grad():
            ### Validation ###
            epoch_val_loss = models.frcnn_evaluate_loss(model, val_loader, device)
            losses_val_NEW.append(epoch_val_loss)
        '''


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
                      'evals': eval_per_n_epoch}


        # Save model state and artifacts after each run and most recent state
        torch.save(model_info, save_path)
        
        model_path_eval = state_per_epoch + 'epoch {}'.format(epoch+1)
        torch.save(model_info, model_path_eval)

        # Save model state in torchscript format for inference 
        torch.save(model, save_path+'.pt')
    
    return losses_train