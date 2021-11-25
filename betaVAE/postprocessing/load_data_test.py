# -*- coding: utf-8 -*-
# /usr/bin/env python3

# Imports
import torch
import os
import pandas as pd
import datasets
import numpy as np


data_dir = '/neurospin/dico/data/deep_folding/current/crops/CINGULATE/mask/sulcus_based/2mm/centered_combined/'

subject_dir = "/neurospin/dico/data/deep_folding/current/"
side = 'R'

def load(data_dir=data_dir, subject_dir=subject_dir):

    ###### HCP Train + val sets
    train_list = pd.read_csv(os.path.join(subject_dir, 'train.csv'), header=None,
                            usecols=[0], names=['subjects'])

    train_list['subjects'] = train_list['subjects'].astype('str')

    tmp = pd.read_pickle(os.path.join(data_dir, "hcp", f"{side}skeleton.pkl")).T
    tmp.index.astype('str')
    tmp = tmp.merge(train_list, left_on = tmp.index, right_on='subjects', how='right')
    filenames = list(train_list['subjects'])
    train_set = datasets.SkeletonDataset(dataframe=tmp, filenames=filenames)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                              shuffle=False, num_workers=0)

    ###### HCP test sets
    test_list = pd.read_csv(os.path.join(subject_dir, 'test.csv'), header=None,
                            usecols=[0], names=['subjects'])

    test_list['subjects'] = test_list['subjects'].astype('str')

    tmp = pd.read_pickle(os.path.join(data_dir, "hcp", f"{side}skeleton.pkl")).T
    tmp.index.astype('str')
    tmp = tmp.merge(test_list, left_on = tmp.index, right_on='subjects', how='right')
    filenames = list(test_list['subjects'])
    test_set = datasets.SkeletonDataset(dataframe=tmp, filenames=filenames)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                              shuffle=False, num_workers=0)

    ###### Tissier database
    labels_set = pd.read_csv('/neurospin/dico/data/deep_folding/datasets/ACC_patterns/tissier_labels.csv')
    dico_labels = {labels_set['Sujet'][k]: 0 if labels_set['ACC_right_2levels_new'][k]=='absent' else 1 for k in range(len(labels_set))}
    df_tissier_labels = pd.DataFrame(data={'sub': list(dico_labels.keys()), 'cing':list(dico_labels.values())})

    tmp2 = pd.read_pickle(os.path.join(data_dir, "tissier_2018", f"{side}skeleton.pkl")).T
    tmp2.index.astype('str')

    # keeping only labeled subjects
    tmp2 = tmp2.merge(df_tissier_labels, left_on=tmp2.index, right_on="sub", how="right")

    filenames2 = list(tmp2['sub'])
    tissier = datasets.SkeletonDataset(dataframe=tmp2, filenames=filenames2)

    labels = np.array(['hcp' for k in range(len(test_list))] \
             + ['tissier' for k in range(len(tmp2))])

    tissier = torch.utils.data.DataLoader(tissier, batch_size=1,
                                              shuffle=False, num_workers=0)

    dico_set_loaders = {'train': train_loader, 'test': test_loader, 'tissier': tissier}

    return dico_set_loaders
