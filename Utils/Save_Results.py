# -*- coding: utf-8 -*-
"""
Created on Thursday April 25 22:32:00 2024
Save aggregate results from saved models
@author: salimalkharsa, jpeeples
"""
## Python standard libraries
from __future__ import print_function
from __future__ import division
import os
import glob
from collections import defaultdict
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import itertools
import json
import pdb

def generate_filename_optuna(Network_parameters, split, trial_number):
    # Generate filename for saving results
    base_path = Network_parameters['folder']
    mode = Network_parameters['mode']
    method = Network_parameters['method']
    dataset = Network_parameters['Dataset']
    split_run = f"Run_{split + 1}"
    trial_dir = f"trial_{trial_number}"

    if mode == 'student':
        model = Network_parameters['student_model']
        if Network_parameters['feature_extraction']:
            filename = f"{base_path}/{mode}/{method}/{dataset}/{model}/{trial_dir}/{split_run}/"
        else:
            filename = f"{base_path}/{mode}/{method}/{dataset}/{model}/{trial_dir}/{split_run}/"
    
    elif mode == 'teacher':
        model = Network_parameters['teacher_model']
        if Network_parameters['use_pretrained']:
            if Network_parameters['feature_extraction']:
                filename = f"{base_path}/{mode}/{Network_parameters['model']}/{method}/{dataset}/{model}/{trial_dir}/{split_run}/"
            else:
                filename = f"{base_path}/{mode}/{Network_parameters['model']}/{method}/{dataset}/{model}/{trial_dir}/{split_run}/"
        else:
            if Network_parameters['feature_extraction']:
                filename = f"{base_path}/{mode}/{method}/{dataset}/{model}/{trial_dir}/{split_run}/"
            else:
                filename = f"{base_path}/{mode}/{method}/{dataset}/{model}/{trial_dir}/{split_run}/"
    
    else: # KD
        student_model = Network_parameters['student_model']
        teacher_model = Network_parameters['teacher_model']
        task_flag = Network_parameters['task_flag']
        if Network_parameters['feature_extraction']:
            filename = f"{base_path}/{mode}/{method}/{dataset}/{student_model}_{teacher_model}/{trial_dir}/{split_run}/"
        else:
            filename = f"{task_flag}/{base_path}/{mode}/{method}/{dataset}/{student_model}_{teacher_model}/{trial_dir}/{split_run}/"
    
    # Create directory if it does not exist
    if not os.path.exists(filename):
        os.makedirs(filename)
    return filename

def generate_filename(Network_parameters,split):
    # Generate filename for saving results

    #Add ability to only train output layer or entire teacher
    if Network_parameters['mode'] == 'student':

        if Network_parameters['feature_extraction'] :
            filename = '{}/{}/{}/{}/{}/Run_{}/'.format(Network_parameters['folder'],
                                                        Network_parameters['mode'], 
                                                        Network_parameters['method'],
                                                        Network_parameters['Dataset'],
                                                        Network_parameters['student_model'],
                                                        split+1)
        else:
            filename = '{}/{}/{}/{}/{}/Run_{}/'.format(Network_parameters['folder'],
                                                        Network_parameters['mode'], 
                                                        Network_parameters['method'],
                                                        Network_parameters['Dataset'],
                                                        Network_parameters['student_model'],
                                                        split+1)
            
        
    
    elif Network_parameters['mode'] == 'teacher':
        if Network_parameters['use_pretrained']: 
            if Network_parameters['feature_extraction'] :
                filename = '{}/{}/{}/{}/{}/{}/Run_{}/'.format(Network_parameters['folder'],
                                                            Network_parameters['mode'], 
                                                            Network_parameters['model'],
                                                            Network_parameters['method'],
                                                            Network_parameters['Dataset'],
                                                            Network_parameters['teacher_model'],
                                                            split+1)
            else:
                filename = '{}/{}/{}/{}/{}/{}/Run_{}/'.format(Network_parameters['folder'],
                                                            Network_parameters['mode'], 
                                                            Network_parameters['model'],
                                                            Network_parameters['method'],
                                                            Network_parameters['Dataset'],
                                                            Network_parameters['teacher_model'],
                                                            split+1)
        else:
            if Network_parameters['feature_extraction'] :
                filename = '{}/{}/{}/{}/{}/Run_{}/'.format(Network_parameters['folder'],
                                                            Network_parameters['mode'], 
                                                            Network_parameters['method'],
                                                            Network_parameters['Dataset'],
                                                            Network_parameters['teacher_model'],
                                                            split+1)
            else:
                filename = '{}/{}/{}/{}/{}/Run_{}/'.format(Network_parameters['folder'],
                                                            Network_parameters['mode'], 
                                                            Network_parameters['method'],
                                                            Network_parameters['Dataset'],
                                                            Network_parameters['teacher_model'],
                                                            split+1)           
    else: #KD

        if Network_parameters['feature_extraction'] :
            filename = '{}/{}/{}/{}/{}_{}/Run_{}/'.format(Network_parameters['folder'],
                                                        Network_parameters['mode'], 
                                                        Network_parameters['method'],
                                                        Network_parameters['Dataset'],
                                                        Network_parameters['student_model'],
                                                        Network_parameters['teacher_model'],
                                                        split+1)
        else:

            filename = '{}/{}/{}/{}/{}_{}/Run_{}/'.format(Network_parameters['folder'],
                                                        Network_parameters['mode'], 
                                                        Network_parameters['method'],
                                                        Network_parameters['Dataset'],
                                                        Network_parameters['student_model'],
                                                        Network_parameters['teacher_model'],
                                                        split+1)
        
    # Create directory if it does not exist
    if not os.path.exists(filename):
        os.makedirs(filename)
    return filename

def plot_avg_confusion_matrix(cm, classes,
                              normalize=False,
                              title='Confusion matrix',
                              cmap=plt.cm.Blues,
                              show_percent=True, ax=None,
                              fontsize=32):
    # Set global font sizes
    plt.rcParams.update({
        'font.size': fontsize,
        'axes.titlesize': fontsize + 2,
        'axes.labelsize': fontsize +2,
        'xtick.labelsize': fontsize + 2,
        'ytick.labelsize': fontsize + 2
    })

    # Compute average CM values
    std_cm = np.int64(np.ceil(np.std(cm, axis=2)))
    cm = np.int64(np.ceil(np.mean(cm, axis=2)))

    # Convert cm to percentages if show_percent is True
    if show_percent:
        cm_percent = 100 * cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_percent_std = 100 * std_cm.astype('float') / (std_cm.sum(axis=1)[:, np.newaxis] + 1e-6)
        display_matrix = cm_percent  # Use the percentage matrix for display
        vmin, vmax = 0, 100  # Set color bar limits for percentage display
    else:
        display_matrix = cm  # Use raw values if not displaying as percentages
        vmin, vmax = None, None  # Color bar limits for raw values

    # Generate new figure if needed
    if ax is None:
        fig, ax = plt.subplots(figsize=(19, 19))  # Adjust figure size as needed

    # Plot the confusion matrix
    im = ax.imshow(display_matrix, interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    plt.title(title)
    cb = plt.colorbar(im, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=fontsize)  # Colorbar font size

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes, fontsize=fontsize+6)  # X-axis labels font size
    ax.set_yticklabels(classes, fontsize=fontsize+6)  # Y-axis labels font size
    ax.set_xlabel('Predicted Label', fontsize=fontsize+6)  # X-axis label font size
    ax.set_ylabel('True Label', fontsize=fontsize+6)  # Y-axis label font size
    ax.set_title(title, fontsize=fontsize + 6)  # Title font size

    fmt = '.2f' if normalize or show_percent else 'd'
    thresh = cm.max() / 2.

    # Display text in each cell
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):

        s = f"{cm[i, j]}±{std_cm[i, j]}"
        ax.text(j, i, s,
                horizontalalignment="center", verticalalignment='center',
                color="white" if cm[i, j] > thresh else "black",
                fontsize=fontsize - 2,)  # Font size for text inside cells

    ax.set_ylim((len(classes) - 0.5, -0.5))
    plt.setp(ax.get_xticklabels(), rotation=45, fontsize=fontsize)
    plt.tight_layout()



def aggregate_tensorboard_logs(root_dir, save_dir, dataset):
    aggregated_results = defaultdict(list)
    # Create save directory if it doesn't exist
    save_dir = '{}/{}'.format(root_dir, save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Define the specific metrics to save
    specific_metrics = [
        'val_accuracy', 
        'test_accuracy', 
        'test_weighted_f1', 
        'test_weighted_precision', 
        'test_weighted_recall'
    ]
    
    # Traverse through the directory structure
    for run_dir in os.listdir(root_dir):
        run_path = os.path.join(root_dir, run_dir)
        if not os.path.isdir(run_path):
            continue

        # Look for event files within each run directory
        event_file = os.path.join(run_path, 'tb_logs', 'model_logs', 'version_0','events.out.tfevents.*')
        event_files = glob.glob(event_file)


        for event_file in event_files:
            event_acc = EventAccumulator(event_file)
            event_acc.Reload()

            # Extract scalar data from event file
            tags = event_acc.Tags()['scalars']

            for metric in specific_metrics:
                if metric in tags:
                    events = event_acc.Scalars(metric)
                    values = [event.value for event in events]
                    aggregated_results[metric].extend(values)

    # Aggregate metrics
    final_aggregated_results = {}
    text_output = []

    for metric, values in aggregated_results.items():
        values = np.array(values) * 100
        mean = np.mean(values)
        std = np.std(values)
        final_aggregated_results[metric] = {'mean': mean, 'std': std}
        text_output.append(f"{metric}: Mean = {mean:.6f}, Std = {std:.6f}")


    # Save results in JSON file
    json_path = "{}/{}_aggregated_specific_metrics.json".format(save_dir, dataset)
    with open(json_path, "w") as outfile: 
        json.dump(final_aggregated_results, outfile)

    # Save results in text file
    text_path = "{}/{}_aggregated_specific_metrics.txt".format(save_dir, dataset)
    with open(text_path, "w") as textfile:
        textfile.write("\n".join(text_output))

    return final_aggregated_results


def aggregate_and_visualize_confusion_matrices(root_dir, save_dir, dataset,
                                               label_names=None, cmap='Blues',
                                               threshold=10, figsize=(12, 12),
                                              fontsize=20, title=False):
    
    # pdb.set_trace()
    # Create save directory if it doesn't exist
    save_dir = '{}/{}'.format(root_dir, save_dir)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        
    aggregated_matrix_list = []
    test_matrix_count = 0

    # Aggregate confusion matrices
    for run_dir in os.listdir(root_dir):
        run_path = os.path.join(root_dir, run_dir)
        if not os.path.isdir(run_path):
            continue

        # Look for CSV files containing confusion matrices
        csv_files = [f for f in os.listdir(run_path) if f.endswith('.csv') and 'confusion_matrix' in f]
        for csv_file in csv_files:
            # Read the test matrix
            if 'test' in csv_file:
                matrix = pd.read_csv(os.path.join(run_path, csv_file), index_col=0).to_numpy()
                aggregated_matrix_list.append(matrix[..., np.newaxis])
                test_matrix_count += 1
    
    # Convert list to numpy array
    aggregated_matrix_list = np.concatenate(aggregated_matrix_list, axis=-1)
    test_mean_matrix = np.mean(aggregated_matrix_list, axis=-1)
    
    if title:
        title_name = '{} Confusion Matrix'.format(dataset)
    else:
        title_name = None


    if label_names is not None and len(label_names) <= threshold:
        plt.figure(figsize=(30,30))
        fig, ax = plt.subplots(figsize=(30,30))
        # pdb.set_trace()
        plot_avg_confusion_matrix(aggregated_matrix_list, label_names, 
                                  title=title_name)
        # Save the heatmap as an image
        plt.savefig(os.path.join(save_dir, '{}_aggregated_confusion_matrix.png'.format(dataset)), bbox_inches='tight')
        plt.close()
        return save_dir
    else:
        plt.figure(figsize=figsize)
        sns.heatmap(test_mean_matrix, cmap=cmap, cbar=True, annot=False)
        plt.title(title_name, fontsize=14)
        plt.xlabel('Predicted Label', fontsize=14)
        plt.ylabel('True Label', fontsize=14)
        # Save the heatmap as an image
        plt.savefig(os.path.join(save_dir, 'aggregated_confusion_matrix_'+dataset+'.png'), bbox_inches='tight')
        plt.close()
        return save_dir

 