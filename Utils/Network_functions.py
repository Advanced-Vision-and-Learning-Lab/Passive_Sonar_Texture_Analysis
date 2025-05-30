# -*- coding: utf-8 -*-
"""
Functions to generate model and train/validate/test
@author: jpeeples
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import numpy as np
import time
import copy

## PyTorch dependencies
import torch
import torch.nn as nn

## Local external libraries
from Utils.Histogram_Model import HistRes
from barbar import Bar
from .pytorchtools import EarlyStopping
from Utils.TDNN import TDNN
from Utils.Feature_Extraction_Layer import Feature_Extraction_Layer



def train_model(model, dataloaders, criterion, optimizer, device,feature_extraction_layer,
                saved_bins=None, saved_widths=None, histogram=True,
                num_epochs=25, scheduler=None, dim_reduced=True):
    since = time.time()

    val_acc_history = []
    train_acc_history = []
    train_error_history = []
    val_error_history = []

    early_stopping = EarlyStopping(patience=10, verbose=True)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_epoch = 0
    best_loss = np.inf
    valid_loss = best_loss
    print('Training Model...')
    

    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)


        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:

            if phase == 'train':
                model.train()  # Set model to training mode 
                feature_extraction_layer.train()
            else:
                model.eval()   # Set model to evaluate mode
                feature_extraction_layer.eval()
            
            running_loss = 0.0
            running_corrects = 0
            # Iterate over data.
            for idx, (inputs, labels, index) in enumerate(Bar(dataloaders[phase])):
                
                inputs = inputs.to(device)

                labels = labels.to(device)
                index = index.to(device)
    
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    
                    #Pass through feature layer 
                    features = feature_extraction_layer(inputs)
                    outputs = model(features)
                    loss = criterion(outputs, labels)
    
                    _, preds = torch.max(outputs, 1)
    
                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
    
                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds.data == labels.data)
        
            epoch_loss = running_loss / (len(dataloaders[phase].sampler))
            epoch_acc = running_corrects.item() / (len(dataloaders[phase].sampler))
            
            if phase == 'train':
                if scheduler is not None:
                    scheduler.step()
                train_error_history.append(epoch_loss)
                train_acc_history.append(epoch_acc)
                if(histogram):
                    if dim_reduced:
                        #save bins and widths
                        saved_bins[epoch+1,:] = model.module.histogram_layer[-1].centers.detach().cpu().numpy()
                        saved_widths[epoch+1,:] = model.module.histogram_layer[-1].widths.reshape(-1).detach().cpu().numpy()
                    else:
                        # save bins and widths
                        saved_bins[epoch + 1, :] = model.module.histogram_layer.centers.detach().cpu().numpy()
                        saved_widths[epoch + 1, :] = model.module.histogram_layer.widths.reshape(
                            -1).detach().cpu().numpy()

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_epoch = epoch
                best_acc = epoch_acc
                valid_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())

            if phase == 'val':
                val_error_history.append(epoch_loss)
                val_acc_history.append(epoch_acc)

            print()
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc)) 
            print()
    
        #Check validation loss
        early_stopping(valid_loss, model)
        
        if early_stopping.early_stop:
            print()
            print("Early stopping")
            print()
            break
     
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print()

    # load best model weights
    model.load_state_dict(best_model_wts)
    
    # Return losses as dictionary
    train_loss = train_error_history
    
    val_loss = val_error_history
 
    #Return training and validation information
    train_dict = {'best_model_wts': best_model_wts, 'val_acc_track': val_acc_history, 
                  'val_error_track': val_loss,'train_acc_track': train_acc_history, 
                  'train_error_track': train_loss,'best_epoch': best_epoch, 
                  'saved_bins': saved_bins, 'saved_widths': saved_widths}
    
    return train_dict

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False
            

    
def test_model(dataloader,model,feature_extraction_layer,criterion,device):
    #Initialize and accumalate ground truth, predictions, and image indices
    GT = np.array(0)
    Predictions = np.array(0)
    Index = np.array(0)
    
    running_corrects = 0
    running_loss = 0.0
    model.eval()
    feature_extraction_layer.eval()
    
    # Iterate over data
    print('Testing Model...')
    with torch.no_grad():
        for idx, (inputs, labels, index) in enumerate(dataloader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            index = index.to(device)
            # Forward pass for logits of network
            features = feature_extraction_layer(inputs)
            outputs = model(features)
            loss = criterion(outputs, labels)
           
            #Get predictions for test data
            _, preds = torch.max(outputs, 1)
    
            #If test, accumulate labels for confusion matrix
            GT = np.concatenate((GT,labels.detach().cpu().numpy()),axis=None)
            Predictions = np.concatenate((Predictions,preds.detach().cpu().numpy()),axis=None)
            Index = np.concatenate((Index,index.detach().cpu().numpy()),axis=None)
            
        
            # Running statistics for classification metrics
            running_corrects += torch.sum(preds == labels.data)
            running_loss += loss.item() * inputs.size(0)
            
    test_loss = running_loss / (len(dataloader.sampler))
    test_acc = running_corrects.item() / (len(dataloader.sampler))
    
    print('Test Accuracy: {:4f}'.format(test_acc))
    
    test_dict = {'GT': GT[1:], 'Predictions': Predictions[1:], 'Index':Index[1:],
                  'test_acc': np.round(test_acc*100,2),
                'test_loss': test_loss}
    
    return test_dict


def initialize_model(model_name, num_classes, in_channels, out_channels,
                     feature_extract=False, histogram=True, histogram_layer=None,
                     parallel=True, use_pretrained=True, add_bn=True, scale=5,
                     feat_map_size=4, TDNN_feats=1, input_feature='STFT', RGB=True,
                     sample_rate=16000, window_length=250, hop_length=64,
                     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],
                     checkpoint_path=None):  # <-- new argument

    
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0
    if(histogram):
        # Initialize these variables which will be set in this if statement. Each of these
        # variables is model specific.
        model_ft = HistRes(histogram_layer,parallel=parallel,
                           model_name=model_name,add_bn=add_bn,scale=scale,
                           pretrained=use_pretrained, TDNN_feats=TDNN_feats)

        set_parameter_requires_grad(model_ft.backbone, feature_extract)
        
        #Reduce number of conv channels from input channels to input 
        #channels/number of bins*feat_map size (2x2)
        reduced_dim = int((out_channels/feat_map_size)/(histogram_layer.numBins))
        
        #If input channels equals reduced/increase, don't apply 1x1 convolution
        if (in_channels==reduced_dim): 
            model_ft.histogram_layer = histogram_layer
        else:
            conv_reduce = nn.Conv2d(in_channels,reduced_dim,(1,1))
            model_ft.histogram_layer = nn.Sequential(conv_reduce,histogram_layer)
        if(parallel):
            num_ftrs = model_ft.fc.in_features*2
        else:
            num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, num_classes)
        input_size = 224
        
    # Baseline model
    else:            
        if model_name == "TDNN": 
            model_ft = TDNN(in_channels=TDNN_feats)
            set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224    
        else:
            raise RuntimeError('{} not implemented'.format(model_name))

    feature_layer = Feature_Extraction_Layer(input_feature=input_feature,sample_rate=sample_rate, window_length=window_length,window_size=512, hop_size=160, 
        mel_bins=64, fmin=50, 
        hop_length=hop_length,fmax=8000, classes_num=527)


    #Return baseline model, desired input size, and feature layer
    return model_ft, input_size, feature_layer


