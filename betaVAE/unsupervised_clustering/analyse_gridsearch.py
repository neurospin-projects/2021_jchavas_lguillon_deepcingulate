# /usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
from operator import itemgetter
import os
import re

import json

root_dir = "/neurospin/dico/lguillon/midl_22/new_design/gridsearch/"
list_n_latent = [2, 5, 15, 20, 40, 75, 100]
kl = [1, 2, 5, 8]
list_loss_val = []
list_silhouette = []

for file in os.listdir(root_dir):
    print(file)
    if 'old' not in file:
        n_latent = re.search('n_(\d{1,2})', file).group(1)
        kl = re.search('kl_(\d{1})', file).group(1)
        print(n_latent, kl)

        for
        f = open(f"{os.path.join(root_dir, file)}/results.json")
        data = json.load(f)

        for silhouette, loss_val in zip(data['AffinityPropagation'], data['loss_val']):
            print(f"AffinityPropagation silhouette score: {silhouette} \n" \
                  f"Final loss val: {loss_val}")
        # Closing file
        f.close()
        list_loss_val.append(loss_val)
        list_silhouette.append(silhouette)

print(len(list_n_latent), len(list_loss_val), len(list_silhouette))
