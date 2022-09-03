import pickle 
import torch
from engine import train_one_epoch, evaluate
import utils
from tqdm import tqdm
import numpy as np


def train(dataset_train, dataset_test, model, epoch_count, eval_freq, dir_out):

    loss_per_epoch_train = []
    loss_per_epoch_val = []
    eval_per_n_epoch = []

    # train on the GPU or on the CPU, if a GPU is not available
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    # our dataset has two classes only - background and person
    #num_classes = 7
    # use our dataset and defined transformations
    dataset = dataset_train
    dataset_test = dataset_test

    '''
    # split the dataset in train and test set
    indices = torch.randperm(len(dataset)).tolist()
    # MODIFIED FOR HAUL TRUCK, there are only 99 haul truck examples
    dataset = torch.utils.data.Subset(dataset, indices[:-20])
    # MODIFIED FOR HAUL TRUCK, there are only 99 haul truck examples
    dataset_test = torch.utils.data.Subset(dataset_test, indices[-20:])
    '''

    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True, num_workers=2,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=2,
        collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = model

    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    #optimizer = torch.optim.SGD(params, lr=0.005,momentum=0.9, weight_decay=0.0005)
    optimizer = torch.optim.Adam(params, lr=0.0001)

    # and a learning rate scheduler
    # TODO lr_scheduler methods
    # TODO, step_size and gamma 90k epoch 10x lr shrink 60k iterations and again at 80k
    #lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)

    # epochs to train for
    num_epochs = epoch_count

    for epoch in range(num_epochs):
        # train for one epoch, printing every 10 iterations
        avg_loss = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        loss_per_epoch_train.append(avg_loss.meters['loss'])






        # calculate validation loss

        val_losses = []
        # Process all data in the data loader 
        for imgs, annotations in tqdm(data_loader_test, desc = 'Validation loss calculation'):
            
            # Prepare images and annotations
            imgs = list(img.to(device) for img in imgs)
            annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
            
            # Calculate loss 
            with torch.no_grad():
                loss_dict = model(imgs, annotations)
                print('loss dict')
                print(loss_dict)
            losses = sum(loss for loss in loss_dict.values())
            val_losses.append(losses.cpu().detach().numpy())

        # Val epoch done
        val_loss = np.mean(val_losses)
        loss_per_epoch_val.append(val_loss)




        if epoch % eval_freq == (eval_freq-1):
            # update the learning rate
            # TODO lr_scheduler methods
            #lr_scheduler.step()
            
            # evaluate on the test dataset
            eval_data = evaluate(model, data_loader_test, device=device)

            eval_per_n_epoch.append(eval_data)

    print(loss_per_epoch_train)
    print(loss_per_epoch_val)


    with open(dir_out + 'loss_per_epoch.pkl', 'wb') as f:
        pickle.dump(loss_per_epoch_train, f)

    with open(dir_out + 'loss_per_epoch_val.pkl', 'wb') as f:
        pickle.dump(loss_per_epoch_val, f)

    with open(dir_out + 'eval_per_n_epoch.pkl', 'wb') as f:
        pickle.dump(eval_per_n_epoch, f)

    # TODO save model
  

    print("TRAINING COMPLETE")