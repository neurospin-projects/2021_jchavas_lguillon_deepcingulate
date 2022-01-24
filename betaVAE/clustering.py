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


import torch
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AffinityPropagation
from sklearn.metrics import silhouette_samples, silhouette_score
import json
import matplotlib.cm as cm
import matplotlib.pyplot as plt


class Cluster():
    """ Performs cluster analysis of encoded subjects
    """
    def __init__(self, X, root_dir):
        self.n_clusters_list = [2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.x = X
        self.dir = root_dir

    def plot_silhouette(self):
        res_silhouette = {'kmeans':{2: 0, 3: 0, 4: 0, 5:0, 6:0, 7: 0, 8: 0, 9:0, 10: 0},
                          'AffinityPropagation':{}}
        for n in self.n_clusters_list:
            cluster_labels= KMeans(n_clusters=n, random_state=0).fit_predict(self.x)
            res_silhouette['kmeans'][n] = str(metrics.silhouette_score(self.x, cluster_labels))

            fig, ax1 = plt.subplots()
            ax1.set_ylim([0, len(self.x) + (n + 1) * 10])
            silhouette_avg = silhouette_score(self.x, cluster_labels)
            print("For n_clusters =", n, "The average silhouette_score with kmeans is :", silhouette_avg)

            # Compute the silhouette scores for each sample
            sample_silhouette_values = silhouette_samples(self.x, cluster_labels)

            y_lower = 10
            for i in range(n):
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

                ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                y_lower = y_upper + 10

            ax1.set_title("The silhouette plot for the various clusters.")
            ax1.set_xlabel("The silhouette coefficient values")
            ax1.set_ylabel("Cluster label")

            ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax1.set_yticks([])
            ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.savefig(f"{self.dir}kmeans_silhouette_{n}clusters.png")

        af = AffinityPropagation(random_state=0, max_iter=1000).fit(self.x)
        cluster_labels_ini = af.labels_
        initial_centers = af.cluster_centers_indices_
        n_clusters_ = len(initial_centers)
        while n_clusters_ > 5:
            af = AffinityPropagation(random_state=0).fit(self.x[af.cluster_centers_indices_])
            center_cluster_labels = af.labels_
            x_cluster_label = af.predict(self.x)
            n_clusters_ = len(af.cluster_centers_indices_)
            print(n_clusters_)

        if n_clusters_>1:
            res_silhouette['AffinityPropagation'][n_clusters_] = str(metrics.silhouette_score(self.x, x_cluster_label))
            fig2, ax2 = plt.subplots()
            ax2.set_ylim([0, len(self.x) + (n + 1) * 10])
            silhouette_avg = silhouette_score(self.x, x_cluster_label)
            print("For n_clusters =", n_clusters_, "The average silhouette_score with AffinityPropagation is :", silhouette_avg)

            sample_silhouette_values = silhouette_samples(self.x, cluster_labels)

            y_lower = 10
            for i in range(n_clusters_):
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

                ax2.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

                y_lower = y_upper + 10

            ax2.set_title("The silhouette plot for the various clusters.")
            ax2.set_xlabel("The silhouette coefficient values")
            ax2.set_ylabel("Cluster label")

            ax2.axvline(x=silhouette_avg, color="red", linestyle="--")

            ax2.set_yticks([])
            ax2.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
            plt.savefig(f"{self.dir}AffinityPropagation_silhouette.png")

        print(res_silhouette)
        return res_silhouette
