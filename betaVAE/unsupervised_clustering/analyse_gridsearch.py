# /usr/bin/env python3
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import os
import re

import json

root_dir = "/neurospin/dico/lguillon/midl_22/new_design/gridsearch/weight11"
list_n_latent = [2, 5, 15, 20, 40, 75, 100]
list_kl = [1, 2, 5]

for kl in list_kl :
    list_loss_val = []
    list_silhouette = []
    for n in list_n_latent:
        file = f"n_{n}_kl_{kl}"
        print(file)

        f = open(f"{os.path.join(root_dir, file)}/results.json")
        data = json.load(f)

        for silhouette, loss_val in zip(list(data['AffinityPropagation'].values()), data['loss_val']):
            print(f"AffinityPropagation silhouette score: {silhouette} \n" \
                 f"Final loss val: {loss_val}")
        # Closing file
        f.close()
        list_loss_val.append(loss_val)
        list_silhouette.append(silhouette)

    print(len(list_loss_val), list_loss_val)
    fig, ax = plt.subplots(figsize=(12,8))
    ax.set_xlabel('Latent space size')
    ax.set_ylabel('Validation loss', color='r')
    ax.plot(list_n_latent, list_loss_val, color='r')
    ax.tick_params(axis='y', labelcolor='r')

    ax2 = ax.twinx()  # instantiate a second axes that shares the same x-axis
    ax2.set_ylabel('Silhouette score', color='b')  # we already handled the x-label with ax1
    ax2.plot(list_n_latent, list_silhouette, color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    #plt.plot(list_n_latent, list_loss_val, 'r')
    #plt.plot(list_n_latent, list_silhouette, 'g')
    plt.savefig(f"{root_dir}/visu_kl_{kl}.png")

print(len(list_n_latent), len(list_loss_val), len(list_silhouette))
