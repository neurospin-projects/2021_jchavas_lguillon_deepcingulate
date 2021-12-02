# -*- coding: utf-8 -*-
# /usr/bin/env python3
#
#  This software and supporting documentation are distributed by
#      Institut Federatif de Recherche 49
#      CEA/NeuroSpin, Batiment 145,
#      91191 Gif-sur-Yvette cedex
#      France
#
# This software is governed by the CeCILL license version 2 under
# French law and abiding by the rules of distribution of free software.
# You can  use, modify and/or redistribute the software under the
# terms of the CeCILL license version 2 as circulated by CEA, CNRS
# and INRIA at the following URL "http://www.cecill.info".
#
# As a counterpart to the access to the source code and  rights to copy,
# modify and redistribute granted by the license, users are provided only
# with a limited warranty  and the software's author,  the holder of the
# economic rights,  and the successive licensors  have only  limited
# liability.
#
# In this respect, the user's attention is drawn to the risks associated
# with loading,  using,  modifying and/or developing or reproducing the
# software by the user in light of its specific status of free software,
# that may mean  that it is complicated to manipulate,  and  that  also
# therefore means  that it is reserved for developers  and  experienced
# professionals having in-depth computer knowledge. Users are therefore
# encouraged to load and test the software's suitability as regards their
# requirements in conditions enabling the security of their systems and/or
# data to be ensured and,  more generally, to use and operate it in the
# same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license version 2 and that you accept its terms.

"""
Tools in order to create pytorch dataloaders
"""
import os
import sys

p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)

import math
import pandas as pd
import numpy as np
import torch
import torchvision.transforms as transforms
from scipy.ndimage import rotate
#from utils import save_results
from datasets import SkeletonDataset

#from .pynet_transforms import *

subject_dir = "/neurospin/dico/data/deep_folding/current/"
data_dir = "/neurospin/dico/data/deep_folding/current/crops/CINGULATE/mask/sulcus_based/2mm/centered_combined/hcp/"


def create_subset(idx_subset=1):
    """
    Creates datasets from HCP data and depending on dataset split of benchmark
    generation (cf anatomist_tools.benchmark_generation module)
    /!\ ONLY DIFFERENCE FROM create_hcp_sets function is only that it creates
    sets from benchmark split.
    IN: data_dir: str, folder in which save the results
        batch_size: int, size of training batches
    OUT: trainset,
         dataset_val_loader,
         dataset_test_loader
    """
    train_list = pd.read_csv(os.path.join(subject_dir,
                             f"HCP_half_{idx_subset}bis.csv"), header=None,
                             usecols=[0], names=['subjects'])
    train_list['subjects'] = train_list['subjects'].astype('str')

    tmp = pd.read_pickle(os.path.join(data_dir, "Rskeleton.pkl")).T
    #tmp = tmp.rename(columns={0:'subjects'})
    tmp.index.astype('str')

    tmp = tmp.merge(train_list, left_on = tmp.index, right_on='subjects', how='right')

    filenames = list(train_list['subjects'])

    subset = SkeletonDataset(dataframe=tmp, filenames=filenames)


    # Data Augmentation application
    #train_set = AugDatasetTransformer(train_set)

    return subset


#create_train_set(data_dir)
