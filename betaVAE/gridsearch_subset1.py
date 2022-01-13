# /usr/bin/env python3
# coding: utf-8
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

import os
import sys

p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)

import numpy as np
import pandas as pd
import json
import itertools
import torch

from vae import ModelTester
from train_bVAE import train_vae
from clustering import Cluster
from load_data import create_subset


_in_shape = (1, 20, 40, 40) # input size with padding

def gridsearch_bVAE_sub1(trainloader, valloader):
    """ Applies a gridsearch to find best hyperparameters configuration (beta
    value=kl and latent space size=n) based on loss value, silhouette score and
    reconstruction abilities

    Args:
        trainloader: torch loader of training data
        valloader: torch loader of validation data
    """

    torch.manual_seed(0)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            vae = nn.DataParallel(vae)
    weights = [1, 2]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights, reduction='sum')

    config = {"kl": [1, 2, 5, 8, 10],
              "n": [2, 5, 15, 20, 40, 75, 100]
    }

    for kl, n in list(itertools.product(config["kl"], config["n"])):
        cur_config = {"kl": kl, "n": n}
        root_dir = f"/neurospin/dico/lguillon/midl_22/new_design/gridsearch/TEST_n_{n}_kl_{kl}/"

        try:
            os.mkdir(root_dir)
        except FileExistsError:
            print("Directory " , root_dir ,  " already exists")
            pass
        print(cur_config)

        """ Train model for given configuration """
        vae, final_loss_val = train_vae(cur_config, _in_shape, trainloader, valloader,
                        root_dir=root_dir)

        """ Evaluate model performances """
        dico_set_loaders = {'train': trainloader, 'val': valloader}

        tester = ModelTester(model=vae, dico_set_loaders=dico_set_loaders,
                             kl_weight=kl, loss_func=criterion, n_latent=n,
                             depth=3)

        results = tester.test()
        encoded = {loader_name:[results[loader_name][k] for k in results[loader_name].keys()] for loader_name in dico_set_loaders.keys()}
        df_encoded = pd.DataFrame()
        df_encoded['latent'] = encoded['train'] + encoded['val']
        X = np.array(list(df_encoded['latent']))

        cluster = Cluster(X, root_dir)
        res = cluster.plot_silhouette()
        res['loss_val'] = final_loss_val

        with open(f"{root_dir}results_test.json", "w") as json_file:
            json_file.write(json.dumps(res, sort_keys=True, indent=4))

def main():
    """ Main function to perform gridsearch on betaVAE
    """
    torch.manual_seed(0)
    root_dir = f"/neurospin/dico/lguillon/midl_22/new_design/gridsearch/"

    """ Load data and generate torch datasets """
    subset1 = create_subset()
    train_set, val_set = torch.utils.data.random_split(subset1,
                            [round(0.8*len(subset1)), round(0.2*len(subset1))])
    trainloader = torch.utils.data.DataLoader(
                  train_set,
                  batch_size=64,
                  num_workers=8,
                  shuffle=True)
    valloader = torch.utils.data.DataLoader(
                val_set,
                batch_size=1,
                num_workers=8,
                shuffle=True)
    print('size of train loader: ', len(trainloader), 'size of val loader: ',
          len(valloader))

    val_label = []
    for _, path in valloader:
        val_label.append(path[0])

    np.savetxt(f"{root_dir}val_label_TEST.csv", np.array(val_label), delimiter =", ", fmt ='% s')

    gridsearch_bVAE_sub1(trainloader, valloader)


if __name__ == '__main__':
    main()
