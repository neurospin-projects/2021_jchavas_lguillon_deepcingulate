# /usr/bin/env python3
# coding: utf-8

import os
import sys

p = os.path.abspath('../')
if p not in sys.path:
    sys.path.append(p)

import numpy as np
import pandas as pd
import json
from operator import itemgetter
from functools import partial
from torchsummary import summary
import itertools
from deep_folding.utils.pytorchtools import EarlyStopping
from sklearn.model_selection import train_test_split

from vae import *
from train_bVAE import train_vae
from clustering import Cluster
from load_data import create_subset


#_in_shape = (1, 32, 80, 72)
_in_shape = (1, 20, 40, 40)


def gridsearch_bVAE_sub1(trainloader, valloader):

    torch.manual_seed(0)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            vae = nn.DataParallel(vae)
    weights = [1, 2]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')

    config = {"kl": [1, 2, 5, 8, 10],
              "n": [2, 5, 15, 20, 40, 75, 100]
    }
    config = {"kl": [8, 10],
              "n": [75, 100]
    }

    for kl, n in list(itertools.product(config["kl"], config["n"])):
        cur_config = {"kl": kl, "n": n}
        root_dir = f"/neurospin/dico/lguillon/midl_22/new_design/gridsearch/n_{n}_kl_{kl}/"

        try:
            os.mkdir(root_dir)
        except FileExistsError:
            print("Directory " , root_dir ,  " already exists")
            pass
        print(cur_config)

        """ Train model for configuration """
        vae, final_loss_val = train_vae(cur_config, _in_shape, trainloader, valloader,
                        root_dir=root_dir)


        """ Evaluate model performances """
        dico_set_loaders = {'train': trainloader, 'val': valloader}
        #dico_set_loaders = {'val': valloader}

        tester = ModelTester(model=vae, dico_set_loaders=dico_set_loaders,
                             loss_func=criterion, kl_weight=kl,
                             n_latent=n, depth=3, root_dir=root_dir)

        results = tester.test()
        encoded = {loader_name:[results[loader_name][k] for k in results[loader_name].keys()] for loader_name in dico_set_loaders.keys()}
        #print(type(encoded['val']))
        df_encoded = pd.DataFrame()
        df_encoded['latent'] = encoded['train'] + encoded['val']
        #df_encoded['latent'] = encoded['val']
        X = np.array(list(df_encoded['latent']))

        cluster = Cluster(X, root_dir)
        res = cluster.plot_silhouette()
        res['loss_val'] = final_loss_val

        with open(f"{root_dir}results.json", "w") as json_file:
            json_file.write(json.dumps(res, sort_keys=True, indent=4))

def main():

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

    gridsearch_bVAE_sub1(trainloader, valloader)


if __name__ == '__main__':
    main()
