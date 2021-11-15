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

""" beta-VAE

"""

######################################################################
# Imports and global variables definitions
######################################################################

import os
from collections import OrderedDict
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stat
import time
import argparse
import torch
from tqdm import tqdm
import pandas as pd
from torch.autograd import Variable
import torch.nn as nn
from torchvision import models
#from torchsummary import summary

from deep_folding.preprocessing import *
from deep_folding.utils.pytorchtools import EarlyStopping
from betaVAE.postprocessing.test_tools import compute_loss, plot_loss

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import balanced_accuracy_score


class VAE(nn.Module):
    def __init__(self, in_shape, n_latent, depth):
        super().__init__()
        self.in_shape = in_shape
        self.n_latent = n_latent
        c,h,w,d = in_shape
        self.depth = depth
        self.z_dim = w//2**depth # receptive field downsampled 2 times
        self.z_dim_x = h//2**depth

        modules_encoder = []
        for step in range(depth):
            in_channels = 1 if step == 0 else out_channels
            out_channels = 16 if step == 0  else 16 * (2**step)
            modules_encoder.append(('conv%s' %step, nn.Conv3d(in_channels, out_channels,
                    kernel_size=3, stride=1, padding=1)))
            modules_encoder.append(('norm%s' %step, nn.BatchNorm3d(out_channels)))
            modules_encoder.append(('LeakyReLU%s' %step, nn.LeakyReLU()))
            modules_encoder.append(('conv%sa' %step, nn.Conv3d(out_channels, out_channels,
                    kernel_size=4, stride=2, padding=1)))
            modules_encoder.append(('norm%sa' %step, nn.BatchNorm3d(out_channels)))
            modules_encoder.append(('LeakyReLU%sa' %step, nn.LeakyReLU()))
        self.encoder = nn.Sequential(OrderedDict(modules_encoder))

        self.z_mean = nn.Linear(64 * self.z_dim**2 * self.z_dim_x , n_latent) # 8000 -> n_latent = 3
        self.z_var = nn.Linear(64 * self.z_dim**2 * self.z_dim_x, n_latent) # 8000 -> n_latent = 3
        self.z_develop = nn.Linear(n_latent, 64 *self.z_dim**2 * self.z_dim_x) # n_latent -> 8000

        modules_decoder = []
        for step in range(depth-1):
            in_channels = out_channels
            out_channels = in_channels // 2
            ini = 1 if step==0 else 0
            modules_decoder.append(('convTrans3d%s' %step, nn.ConvTranspose3d(in_channels,
                        out_channels, kernel_size=2, stride=2, padding=0, output_padding=(ini,0,0))))
            modules_decoder.append(('normup%s' %step, nn.BatchNorm3d(out_channels)))
            modules_decoder.append(('ReLU%s' %step, nn.ReLU()))
            modules_decoder.append(('convTrans3d%sa' %step, nn.ConvTranspose3d(out_channels,
                        out_channels, kernel_size=3, stride=1, padding=1)))
            modules_decoder.append(('normup%sa' %step, nn.BatchNorm3d(out_channels)))
            modules_decoder.append(('ReLU%sa' %step, nn.ReLU()))
        modules_decoder.append(('convtrans3dn', nn.ConvTranspose3d(16, 1, kernel_size=2,
                        stride=2, padding=0)))
        modules_decoder.append(('conv_final', nn.Conv3d(1, 2, kernel_size=1, stride=1)))
        self.decoder = nn.Sequential(OrderedDict(modules_decoder))
        self.weight_initialization()

    def weight_initialization(self):
        """
        Initializes model parameters according to Gaussian Glorot initialization
        """
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose3d) or isinstance(module, nn.Conv3d):
                #print('weight module conv')
                nn.init.xavier_normal_(module.weight)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                #print('weight module batchnorm')
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)

    def sample_z(self, mean, logvar):
        device = torch.device("cuda", index=0)
        stddev = torch.exp(0.5 * logvar)
        noise = Variable(torch.randn(stddev.size(), device=device))
        return (noise * stddev) + mean

    def encode(self, x):
        #print(x.shape)
        x = self.encoder(x)
        #print(x.shape)
        x = nn.functional.normalize(x, p=2)
        x = x.view(x.size(0), -1)
        #print(x.shape)
        mean = self.z_mean(x)
        #print(mean.shape)
        var = self.z_var(x)
        return mean, var

    def decode(self, z):
        #print("z", z.shape)
        out = self.z_develop(z)
        #print(out.shape)
        out = out.view(z.size(0), 16 * 2**(self.depth-1), self.z_dim_x, self.z_dim, self.z_dim)
        #print(out.shape)
        out = self.decoder(out)
        #print(out.shape)
        return out

    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.sample_z(mean, logvar)
        out = self.decode(z)
        return out, mean, logvar, z


def vae_loss(output, input, mean, logvar, loss_func, kl_weight):
    recon_loss = loss_func(output, input)
    #recon_loss = recon_loss /512000
    """kl_loss = torch.mean(0.5 * torch.sum(
                        torch.exp(logvar) + mean**2 - 1. - logvar, 1),
                        dim=0)"""
    kl_loss = -0.5 * torch.sum(-torch.exp(logvar) - mean**2 + 1. + logvar)
    #kl_loss = kl_loss / 100
    return recon_loss, kl_loss, recon_loss + kl_weight * kl_loss


class classifier(nn.Module):
    def __init__(self, z, n_classes=2):
        super().__init__()
        self.clf = nn.Sequential(
                    nn.Linear(z.shape[1], z.shape[1]),
                    nn.Linear(z.shape[1], n_classes),
                    nn.Sigmoid()
                    )

    def forward(self, z):
        y_pred = self.clf(z)
        return y_pred

    def loss_clf(self, y_pred, y_true):
        loss = nn.BCELoss()
        output_clf = loss(y_pred, y_true)
        return output_clf

class clfTrainer():
    def __init__(self, encoded, label):
        self.encoded = torch.from_numpy(encoded)
        y_true = []
        for i, val in enumerate(label):
            if val=='benchmark':
                y_true.append([1, 0])
            else:
                y_true.append([0, 1])
        #label[label=='benchmark']=(1, 0)
        #label[label=='normal_test']=(0, 1)
        label = torch.from_numpy(np.array(y_true)).float()
        self.label = label
        self.clf = classifier(self.encoded)
        self.optimizer = torch.optim.Adam(self.clf.parameters(), lr=1e-4)

    def train(self):
        for epoch in range(20):
            self.clf.train()
            for z, y_true in zip(self.encoded, self.label):
                self.optimizer.zero_grad()
                output_clf = self.clf(z)
                loss_clf = self.clf.loss_clf(output_clf, y_true)
            loss_clf.backward()
            self.optimizer.step()

    def test(self):
        self.clf.eval()
        loss_tot = []
        total_pred, total_true = [], []
        for z, y_true in zip(self.encoded, self.label):
            self.optimizer.zero_grad()
            output_clf = self.clf(z)
            loss_clf = self.clf.loss_clf(output_clf, y_true)
            class_pred = output_clf.argmax()
            class_true = y_true.argmax()
            total_pred.append(int(class_pred))
            total_true.append(int(class_true))
            loss_tot.append(loss_clf.item())
        return(balanced_accuracy_score(total_true, total_pred))


class ModelTrainer():

    def __init__(self, model, train_loader, val_loader, loss_func, nb_epoch, optimizer, kl_weight,
                n_latent, depth, skeleton, root_dir):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_func = loss_func
        self.nb_epoch = nb_epoch
        self.optimizer = optimizer
        self.kl_weight = kl_weight
        #self.lr = lr
        self.n_latent = n_latent
        self.depth = depth
        self.skeleton = skeleton
        self.root_dir = root_dir

    def train(self):
        id_arr, phase_arr, input_arr, output_arr = [], [], [], []
        conv1, conv2, conv3 = [], [], []
        self.list_loss_train, self.list_loss_val = [], []
        self.list_recon_train, self.list_recon_val = [], []
        self.list_kl_train, self.list_kl_val = [], []
        device = torch.device("cuda", index=0)
        early_stopping = EarlyStopping(patience=20, verbose=True)

        print('skeleton', self.skeleton)

        for epoch in range(self.nb_epoch):
            loss_tot_train, loss_tot_val = 0, 0
            recon_loss_tot, kl_tot, recon_loss_tot_val, kl_tot_val = 0, 0, 0, 0
            self.model.train()
            for inputs, path in self.train_loader:
                self.optimizer.zero_grad()
                #print('input', np.unique(inputs))
                inputs = Variable(inputs).to(device, dtype=torch.float32)
                output, mean, logvar, z = self.model(inputs)
                if self.skeleton:
                    target = torch.squeeze(inputs, dim=1).long()
                    recon_loss, kl, loss_train = vae_loss(output, target, mean,
                                                 logvar, self.loss_func,
                                                 kl_weight=self.kl_weight)
                    output = torch.argmax(output, dim=1)
                else:
                    loss_train = vae_loss(output, inputs, mean, logvar, self.loss_func,
                                      kl_weight=self.kl_weight)
                loss_tot_train += loss_train.item()
                recon_loss_tot += recon_loss.item()
                kl_tot += kl.item()


                #self.optimizer.zero_grad()
                loss_train.backward()
                self.optimizer.step()

            if epoch == self.nb_epoch-1:
                phase = 'train'
                for k in range(len(path)):
                    id_arr.append(path[k])
                    phase_arr.append(phase)
                    input_arr.append(np.array(np.squeeze(inputs[k]).cpu().detach().numpy()))
                    output_arr.append(np.squeeze(output[k]).cpu().detach().numpy())

            self.model.eval()
            for inputs, path in self.val_loader:
                #print(inputs.shape)
                inputs = Variable(inputs).to(device, dtype=torch.float32)

                output, mean, logvar, z = self.model(inputs)
                if self.skeleton:
                    target = torch.squeeze(inputs, 0).long()
                    recon_loss_val, kl_val, loss_val = vae_loss(output, target,
                                                       mean, logvar, self.loss_func,
                                                       kl_weight=self.kl_weight)
                    output = torch.argmax(output, dim=1)
                else:
                    loss_val = vae_loss(output, inputs, mean, logvar, self.loss_func,
                                    self.kl_weight)
                loss_tot_val += loss_val.item()
                recon_loss_tot_val += recon_loss_val.item()
                kl_tot_val += kl_val.item()

            self.list_loss_train.append(loss_tot_train/len(self.train_loader))
            self.list_loss_val.append(loss_tot_val/len(self.val_loader))
            self.list_recon_train.append(recon_loss_tot/len(self.train_loader))
            self.list_recon_val.append(recon_loss_tot_val/len(self.val_loader))
            self.list_kl_train.append(kl_tot/len(self.train_loader))
            self.list_kl_val.append(kl_tot_val/len(self.val_loader))

            if epoch == self.nb_epoch-1:
                phase = 'val'
                for k in range(len(path)):
                    id_arr.append(path[k])
                    phase_arr.append(phase)
                    input_arr.append(np.array(np.squeeze(inputs[k]).cpu().detach().numpy()))
                    output_arr.append(np.squeeze(output[k]).cpu().detach().numpy())

            print('epoch [{}/{}], loss_train:{:.4f}, loss_val:{:.4f}, '.format(epoch,
            self.nb_epoch, loss_tot_train/len(self.train_loader), loss_tot_val/len(self.val_loader)))
            print('loss_recon_train:{:.4f}, loss_recon_val:{:.4f}, '.format(
            recon_loss_tot/len(self.train_loader), recon_loss_tot_val/len(self.val_loader)))
            print('loss_kl_train:{:.4f}, loss_kl_val:{:.4f}, '.format(
            kl_tot/len(self.train_loader), kl_tot_val/len(self.val_loader)))

            if early_stopping.early_stop:
                print("EarlyStopping")
                phase = 'val'
                for k in range(len(path)):
                    id_arr.append(path[k])
                    phase_arr.append(phase)
                    input_arr.append(np.array(np.squeeze(inputs[k]).cpu().detach().numpy()))
                    output_arr.append(np.squeeze(output[k]).cpu().detach().numpy())

        for key, array in {'input': input_arr, 'output' : output_arr,
                'phase': phase_arr, 'id': id_arr}.items():
                    np.save(self.root_dir+key, np.array([array]))

        plot_loss(self.list_loss_train[1:], self.root_dir+'tot_train_')
        plot_loss(self.list_loss_val[1:], self.root_dir+'tot_val_')
        plot_loss(self.list_recon_train[1:], self.root_dir+'recon_train_')
        plot_loss(self.list_recon_val[1:], self.root_dir+'recon_train_')
        plot_loss(self.list_kl_train[1:], self.root_dir+'kl_train_')
        plot_loss(self.list_kl_val[1:], self.root_dir+'kl_val_')
        return min(self.list_loss_train), min(self.list_loss_val), id_arr, phase_arr, input_arr, output_arr


class ModelTester():

    def __init__(self, model, dico_set_loaders, loss_func, kl_weight,
                n_latent, depth, skeleton, root_dir):
        self.model = model
        self.dico_set_loaders = dico_set_loaders
        self.loss_func = loss_func
        self.kl_weight = kl_weight
        self.n_latent = n_latent
        self.depth = depth
        self.skeleton = skeleton
        self.root_dir = root_dir


    def test(self):
        id_arr, input_arr, phase_arr, output_arr = [], [], [], []
        self.list_loss_train, self.list_loss_val = [], []
        device = torch.device("cuda", index=0)

        results = {k:{} for k in self.dico_set_loaders.keys()}

        for loader_name, loader in self.dico_set_loaders.items():
            self.model.eval()
            with torch.no_grad():
                for inputs, path in loader:
                    inputs = Variable(inputs).to(device, dtype=torch.float32)

                    output, mean, logvar, z = self.model(inputs)

                    if self.skeleton:
                        target = torch.squeeze(inputs, dim=0).long()
                        recon_loss_val, kl_val, loss_val = vae_loss(output, target, mean, logvar, self.loss_func,
                                         kl_weight=self.kl_weight)
                        output = torch.argmax(output, dim=1)

                    else:
                        print(output.shape, inputs.shape)
                        loss = vae_loss(output, inputs, mean, logvar, self.loss_func,
                                              kl_weight=self.kl_weight)

                    results[loader_name][path] = (loss_val.item(), output, inputs, np.array(torch.squeeze(z, dim=0).cpu().detach().numpy()))

                    for k in range(len(path)):
                        id_arr.append(path)
                        input_arr.append(np.array(np.squeeze(inputs).cpu().detach().numpy()))
                        output_arr.append(np.squeeze(output).cpu().detach().numpy())
                        phase_arr.append(loader_name)

        for key, array in {'input': input_arr, 'output' : output_arr,
                            'phase': phase_arr, 'id': id_arr}.items():
            np.save(self.root_dir+key+'val', np.array([array]))
        return results
