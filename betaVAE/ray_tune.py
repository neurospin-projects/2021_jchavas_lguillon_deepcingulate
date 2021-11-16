# /usr/bin/env python3
# coding: utf-8

import numpy as np
import pandas as pd
from operator import itemgetter
from functools import partial

from vae import *
import datasets
from deep_folding.utils.pytorchtools import EarlyStopping

from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler

from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn import metrics


def train_vae(config, checkpoint_dir=None, data_dir=None):
    vae = VAE((1, 12, 48, 48), config["n"], depth=3)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            vae = nn.DataParallel(vae)
    vae.to(device)

    weights = [2, 7]
    class_weights = torch.FloatTensor(weights).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')
    optimizer = torch.optim.Adam(vae.parameters(), lr=config["lr"])

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint"))
        vae.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    trainset = datasets.create_train_set()


    kfold = KFold(n_splits=5, shuffle=True)
    for fold, (train_ids, test_ids) in enumerate(kfold.split(trainset)):
        fold_loss = []
        kmean_latent = []
        kmean_tsne = []

        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
        trainloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=64,
            num_workers=8,
            sampler=train_subsampler)
        valloader = torch.utils.data.DataLoader(
            trainset,
            batch_size=8,
            num_workers=8,
            sampler=test_subsampler)

        for epoch in range(200):  # loop over the dataset multiple times
            #print(epoch)
            running_loss = 0.0
            epoch_steps = 0
            for inputs, path in trainloader:
                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                inputs = Variable(inputs).to(device, dtype=torch.float32)
                output, mean, logvar, z = vae(inputs)
                target = torch.squeeze(inputs, dim=1).long()
                recon_loss, kl, loss = vae_loss(output, target, mean,
                                    logvar, criterion,
                                    kl_weight=config["kl"])
                output = torch.argmax(output, dim=1)
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
                epoch_steps += 1
                #print("[%d] loss: %.3f" % (epoch + 1,
                #                            running_loss / epoch_steps))
                running_loss = 0.0
            # Validation loss
            val_loss = 0.0
            val_steps = 0
            total = 0
            encoding = {}
            vae.eval()
            for inputs, path in valloader:
                with torch.no_grad():
                    inputs = Variable(inputs).to(device, dtype=torch.float32)
                    output, mean, logvar, z = vae(inputs)
                    target = torch.squeeze(inputs).long()
                    recon_loss_val, kl_val, loss = vae_loss(output, target,
                                        mean, logvar, criterion,
                                        kl_weight=config["kl"])
                    output = torch.argmax(output, dim=1)
                    encoding[path] = np.array(torch.squeeze(z, dim=0).cpu().detach().numpy())
                    val_loss += loss.cpu().numpy()
                    val_steps += 1
            fold_loss.append(val_loss / val_steps)

        # Evaluation of clustering quality
        encoded = [encoding[k][0] for k in encoding.keys()]
        #print(encoded)
        X = encoded
        #X = np.array(encoding)
        # On latent space
        kmeans = KMeans(n_clusters=2, random_state=0).fit_predict(X)
        kmean_latent.append(metrics.silhouette_score(X, kmeans))
        # On a reduced latent space
        X_embedded = TSNE(n_components=2).fit_transform(X)
        kmeans3d = KMeans(n_clusters=2, random_state=0).fit_predict(X_embedded)
        kmean_tsne.append(metrics.silhouette_score(X_embedded, kmeans3d))

    fold_average = sum(fold_loss)/5
    checkpoint_dir = '/volatile/lg261972'
    with tune.checkpoint_dir(epoch_steps) as checkpoint_dir:
        path = os.path.join(checkpoint_dir, "checkpoint")
        torch.save((vae.state_dict(), optimizer.state_dict()), path)

    # test on benchmark discrimination abilities
    tune.report(loss=fold_average, std=np.std(fold_loss), kmean_latent=np.mean(kmean_latent),
                std_latent=np.std(kmean_latent), kmean_tsne=np.mean(kmean_tsne),
                std_tsne=np.std(kmean_tsne), epoch=epoch)
    print("Finished Training")


def main(num_samples, max_num_epochs, gpus_per_trial):
    data_dir = '/neurospin/dico/lguillon/midl_22/experiments/'
    checkpoint_dir = '/neurospin/dico/lguillon/midl_22/ray_results/'
    trainset = datasets.create_train_set()
    print(len(trainset))

    config = {
              "lr": tune.choice([1e-4]),
              "kl": tune.choice([5, 6, 10]),
              "n": tune.choice([25, 50, 75, 100])
    }
    scheduler = ASHAScheduler(
        metric="loss",
        mode="min",
        max_t=max_num_epochs,
        grace_period=1,
        reduction_factor=2)
    reporter = CLIReporter(
        # parameter_columns=["lr", "kl", "n"],
        metric_columns=["loss", "std", "kmean_latent", "std_latent",
                        "kmean_tsne", "std_tsne", "epoch"])
    result = tune.run(
        partial(train_vae, data_dir=data_dir, checkpoint_dir=checkpoint_dir),
        resources_per_trial={"cpu": 48, "gpu": gpus_per_trial},
        config=config,
        num_samples=num_samples,
        scheduler=scheduler,
        progress_reporter=reporter,
        local_dir='/neurospin/dico/lguillon/midl_22/ray_results')

    best_trial = result.get_best_trial("kmean_latent", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print("Best trial final validation loss: {}".format(
        best_trial.last_result["loss"]))
    print("Best trial kmean_latent validation loss: {}".format(
        best_trial.last_result["kmean_latent"]))

    best_trained_model = VAE((1, 12, 48, 48), best_trial.config["n"], depth=3)
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if gpus_per_trial > 1:
            best_trained_model = nn.DataParallel(best_trained_model)
    best_trained_model.to(device)

    best_checkpoint_dir = best_trial.checkpoint.value
    model_state, optimizer_state = torch.load(os.path.join(
        best_checkpoint_dir, "checkpoint"))
    best_trained_model.load_state_dict(model_state)

    #test_acc = test_accuracy(test_set, best_trained_model, best_trial.config, device)
    #print("Best trial test set accuracy: {}".format(test_acc))


if __name__=='__main__':
    main(100, 200, 1)
