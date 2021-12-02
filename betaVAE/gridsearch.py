# /usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
from operator import itemgetter
from functools import partial
from torchsummary import summary
import itertools

from vae import *
import datasets
#from ray_tune import test_benchmarks
from deep_folding.utils.pytorchtools import EarlyStopping
from sklearn.model_selection import train_test_split

_in_shape = (1, 32, 80, 72)


def train_vae(config, root_dir=None):
    torch.manual_seed(0)
    lr = 2e-4
    vae = VAE(_in_shape, config["n"], depth=3)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            vae = nn.DataParallel(vae)
    vae.to(device)
    summary(vae, _in_shape)

    #weights = [1, 200, 27, 356]
    #weights = [1, 20, 10, 30]
    weights = [1, 1]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')
    optimizer = torch.optim.Adam(vae.parameters(), lr=lr)

    trainset = datasets.create_train_set()
    print(len(trainset))
    #train_set, val_set, _, _ = train_test_split(trainset, labels, test_size=0.33,
    #                                    random_state=42)
    train_set, val_set = torch.utils.data.random_split(trainset,
                            [round(0.8*len(trainset)), round(0.2*len(trainset))])

    trainloader = torch.utils.data.DataLoader(
                  train_set,
                  batch_size=64,
                  num_workers=8,
                  shuffle=True)
    valloader = torch.utils.data.DataLoader(
                val_set,
                batch_size=8,
                num_workers=8,
                shuffle=True)
    print('size of train loader: ', len(trainloader)*64, 'size of val loader: ',
          len(valloader)*8)
    nb_epoch = 500
    early_stopping = EarlyStopping(patience=12, verbose=True, root_dir=root_dir)

    list_loss_train, list_loss_val = [], []
    id_arr, phase_arr, input_arr, output_arr = [], [], [], []
    for epoch in range(nb_epoch):  # loop over the dataset multiple times
        #print(epoch)
        running_loss = 0.0
        epoch_steps = 0
        for inputs, path in trainloader:
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            inputs = Variable(inputs).to(device, dtype=torch.float32)
            target = torch.squeeze(inputs, dim=1).long()
            output, mean, logvar, z = vae(inputs)
            recon_loss, kl, loss = vae_loss(output, target, mean,
                                    logvar, criterion,
                                    kl_weight=config["kl"])
            output = torch.argmax(output, dim=1)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            epoch_steps += 1
        print("[%d] loss: %.3f" % (epoch + 1,
                                        running_loss / epoch_steps))
        list_loss_train.append(running_loss / epoch_steps)
        running_loss = 0.0

        if epoch == nb_epoch-1:
            for k in range(len(path)):
                id_arr.append(path[k])
                phase_arr.append('train')
                input_arr.append(np.array(np.squeeze(inputs[k]).cpu().detach().numpy()))
                output_arr.append(np.squeeze(output[k]).cpu().detach().numpy())

        # Validation loss
        val_loss = 0.0
        val_steps = 0
        total = 0
        vae.eval()
        for inputs, path in valloader:
            with torch.no_grad():
                inputs = Variable(inputs).to(device, dtype=torch.float32)
                output, mean, logvar, z = vae(inputs)
                target = torch.squeeze(inputs, dim=1).long()
                recon_loss_val, kl_val, loss = vae_loss(output, target,
                                        mean, logvar, criterion,
                                        kl_weight=config["kl"])
                output = torch.argmax(output, dim=1)

                val_loss += loss.cpu().numpy()
                val_steps += 1
        valid_loss = val_loss / val_steps
        print("[%d] validation loss: %.3f" % (epoch + 1, valid_loss))
        list_loss_val.append(valid_loss)

        if epoch == nb_epoch-1:
            for k in range(len(path)):
                id_arr.append(path[k])
                phase_arr.append('val')
                input_arr.append(np.array(np.squeeze(inputs[k]).cpu().detach().numpy()))
                output_arr.append(np.squeeze(output[k]).cpu().detach().numpy())

        # early_stopping needs the validation loss to check if it has decresed,
        # and if it has, it will make a checkpoint of the current model
        early_stopping(valid_loss, vae)
        if early_stopping.early_stop:
            print("EarlyStopping")
            for k in range(len(path)):
                id_arr.append(path[k])
                phase_arr.append('val')
                input_arr.append(np.array(np.squeeze(inputs[k]).cpu().detach().numpy()))
                output_arr.append(np.squeeze(output[k]).cpu().detach().numpy())
            break
    for key, array in {'input': input_arr, 'output' : output_arr,
                           'phase': phase_arr, 'id': id_arr}.items():
        np.save(root_dir+key, np.array([array]))

    plot_loss(list_loss_train[1:], root_dir+'tot_train_')
    plot_loss(list_loss_val[1:], root_dir+'tot_val_')

    torch.save((vae.state_dict(), optimizer.state_dict()), root_dir + 'vae.pt')

    # test on benchmark discrimination abilities
    #knn, logreg, svm = test_benchmarks(testset, config, vae)
    #print(knn, logreg, svm)
    print("Finished Training")


def main():
    config = {"kl": [1, 2, 5, 8, 10],
              "n": [5, 15, 40, 75, 100]
    }
    #config = {"kl": [2], "n": [150]}
    for kl, n in list(itertools.product(config["kl"], config["n"])):
        cur_config = {"kl": kl, "n": n}
        root_dir = f"/neurospin/dico/lguillon/midl_22/1mm/n_{n}_kl_{kl}/"

        try:
            os.mkdir(root_dir)
        except FileExistsError:
            print("Directory " , root_dir ,  " already exists")
            pass
        print(cur_config)
        train_vae(cur_config, root_dir=root_dir)


if __name__ == '__main__':
    main()
