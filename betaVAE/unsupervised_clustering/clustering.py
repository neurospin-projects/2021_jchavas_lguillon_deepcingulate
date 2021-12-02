# -*- coding: utf-8 -*-
# /usr/bin/env python3

# Imports
import torch
import pandas as pd
#from preprocessing import create_aims_set
from vae import *
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.metrics import silhouette_samples, silhouette_score
import datasets
#from postprocessing import load_data_test
import json

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np


def cluster():
    """

    """


class Cluster():

    def __init__(self, X, root_dir):
        self.n_clusters_list = [2, 3, 4]
        self.x = X
        self.dir = root_dir

    def plot_silhouette(self):
        """
        """
        res_silhouette = {'kmeans':{2: 0, 3: 0, 4: 0},
                          'spectral':{2: 0, 3: 0, 4: 0}}
        for n in self.n_clusters_list:
            cluster_labels= KMeans(n_clusters=n, random_state=0).fit_predict(self.x)
            res_silhouette['kmeans'][n] = str(metrics.silhouette_score(self.x, cluster_labels))

            fig, ax1 = plt.subplots()
            # The (n_clusters+1)*10 is for inserting blank space between silhouette
            # plots of individual clusters, to demarcate them clearly.
            ax1.set_ylim([0, len(self.x) + (n + 1) * 10])
            silhouette_avg = silhouette_score(self.x, cluster_labels)
            print("For n_clusters =", n, "The average silhouette_score with kmeans is :", silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(self.x, cluster_labels)

            y_lower = 10
            for i in range(n):
                # Aggregate the silhouette scores for samples belonging to
                # cluster i, and sort them
                ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                ith_cluster_silhouette_values.sort()

                size_cluster_i = ith_cluster_silhouette_values.shape[0]
                y_upper = y_lower + size_cluster_i

                color = cm.nipy_spectral(float(i) / n)
                ax1.fill_betweenx(
                    np.arange(y_lower, y_upper),
                    0,
                    ith_cluster_silhouette_values,
                    facecolor=color,
                    edgecolor=color,
                    alpha=0.7,
                )

                # Label the silhouette plots with their cluster numbers at the middle
                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                # Compute the new y_lower for next plot
                y_lower = y_upper + 10  # 10 for the 0 samples

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            # The vertical line for average silhouette score of all the values
            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])  # Clear the yaxis labels / ticks
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.savefig(f"{self.dir}kmeans_silhouette_{n}clusters.png")

            cluster_labels = SpectralClustering(n, n_init=100,
                        assign_labels='discretize').fit_predict(self.x)

            if len(np.unique(np.array(cluster_labels))) > 1:
                res_silhouette['spectral'][n] = str(metrics.silhouette_score(self.x, cluster_labels))
                fig2, ax2 = plt.subplots()
                # The (n_clusters+1)*10 is for inserting blank space between silhouette
                # plots of individual clusters, to demarcate them clearly.
                ax2.set_ylim([0, len(self.x) + (n + 1) * 10])
                silhouette_avg = silhouette_score(self.x, cluster_labels)
                print("For n_clusters =", n, "The average silhouette_score with spectral is :", silhouette_avg)

                # Compute the silhouette scores for each sample
                sample_silhouette_values = silhouette_samples(self.x, cluster_labels)

                y_lower = 10
                for i in range(n):
                    # Aggregate the silhouette scores for samples belonging to
                    # cluster i, and sort them
                    ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]

                    ith_cluster_silhouette_values.sort()

                    size_cluster_i = ith_cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i

                    color = cm.nipy_spectral(float(i) / n)
                    ax2.fill_betweenx(
                        np.arange(y_lower, y_upper),
                        0,
                        ith_cluster_silhouette_values,
                        facecolor=color,
                        edgecolor=color,
                        alpha=0.7,
                    )

                    # Label the silhouette plots with their cluster numbers at the middle
                    ax2.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                    # Compute the new y_lower for next plot
                    y_lower = y_upper + 10  # 10 for the 0 samples

                ax2.set_title("The silhouette plot for the various clusters.")
                ax2.set_xlabel("The silhouette coefficient values")
                ax2.set_ylabel("Cluster label")

                # The vertical line for average silhouette score of all the values
                ax2.axvline(x=silhouette_avg, color="red", linestyle="--")

                ax2.set_yticks([])  # Clear the yaxis labels / ticks
                ax2.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
                plt.savefig(f"{self.dir}spectral_silhouette_{n}clusters.png")
            else:
                res_silhouette['spectral'][n] = 0

        print(res_silhouette)
        return res_silhouette