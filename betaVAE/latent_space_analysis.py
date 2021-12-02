# -*- coding: utf-8 -*-
# /usr/bin/env python3

# Imports
import torch
import pandas as pd
#from preprocessing import create_aims_set
from vae import *
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import datasets
from postprocessing import load_data_test
import json
import umap

"""============== Parameters ============="""
n = 40
kl = 8
side='R'

data_dir = '/neurospin/dico/data/deep_folding/current/crops/CINGULATE/mask/sulcus_based/1mm/hcp/centered_combined/'
root_dir = f"/neurospin/dico/lguillon/midl_22/1mm/n_{n}_kl_{kl}/"
subject_dir = "/neurospin/dico/data/deep_folding/current/"

result = {"silhouette_kmeans_latent_space_hcp":0, # kmeans trained on train + val
          "silhouette_kmeans_tsne_space_hcp":0, # kmeans and tsne fit on train + val
          "silhouette_kmeans_umap_space_hcp":0, # kmeans and tsne fit on train + val
          "silhouette_kmeans_latent_space_tissier":0, # kmeans fit on train + val
          "silhouette_kmeans_tsne_space_tissier":0, # kmeans and tsne fit on train + val
          "accuracy_kmeans_tsne_space_tissier":0, # kmeans fit on train + val
          "accuracy_kmeans_latent_space_tissier":0,
          "ari_kmeans_tsne_space_tissier":0,
          "ari_kmeans_latent_space_tissier":0
           }

"""=================== Loading of the model ================"""
model = VAE((1, 32, 80, 72), n, depth=3)
model.load_state_dict(torch.load(root_dir + 'checkpoint.pt'))


"""=================== Loading of data ================"""
dico_set_loaders = load_data_test.load()

"""=================== Encoding of data ================"""
device = torch.device("cuda", index=0)
model = model.to(device)

class_weights = torch.FloatTensor([1, 1]).to(device)
distance = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')

tester = ModelTester(model=model, dico_set_loaders=dico_set_loaders,
                     loss_func=distance, kl_weight=kl, n_latent=n,
                     depth=3, skeleton=True, root_dir=root_dir)

results = tester.test()

losses = {loader_name:[results[loader_name][k][0] for k in results[loader_name].keys()] for loader_name in dico_set_loaders.keys()}
encoded = {loader_name:[results[loader_name][k][3] for k in results[loader_name].keys()] for loader_name in dico_set_loaders.keys()}

"""================== Analysis on train + val HCP data ==================="""
df_encoded = pd.DataFrame()
df_encoded['latent'] = encoded['train']
X_train = np.array(list(df_encoded['latent']))

#### On latent space
kmeans_hcp= KMeans(n_clusters=2, random_state=0).fit(X_train)
clusters = kmeans_hcp.predict(X_train)
result["silhouette_kmeans_latent_space_hcp"] = str(metrics.silhouette_score(X_train, clusters))

#### On tSNE space
## tSNE reduction
X_embedded = TSNE(n_components=2, perplexity=8).fit_transform(X_train)

x = [X_embedded[k][0] for k in range(len(X_embedded))]
y = [X_embedded[k][1] for k in range(len(X_embedded))]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x, y)

plt.savefig(root_dir+'train_tsne.png')

## kmeans
kmeans_tsne_hcp= KMeans(n_clusters=2, random_state=0).fit(X_embedded)
clusters_tsne = kmeans_tsne_hcp.predict(X_embedded)
result["silhouette_kmeans_tsne_space_hcp"] = str(metrics.silhouette_score(X_embedded, clusters_tsne))

#### On UMAP space
"""reducer = umap.UMAP()
reducer_hcp = reducer.fit(X_train)
X_umap = reducer_hcp.transform(X_train)

kmeans_umap_hcp= KMeans(n_clusters=2, random_state=0).fit(X_umap)
clusters_umap = kmeans_umap_hcp.predict(X_umap)
result["silhouette_kmeans_umap_space_hcp"] = str(metrics.silhouette_score(X_umap, clusters_tsne))
"""

"""=================== Test data: Tissier ================"""
####### TSNE and kmeans ON TEST SET
df_test = pd.DataFrame()
df_test['latent'] = encoded['tissier']
X_tissier = np.array(list(df_test['latent']))

clusters_tissier = kmeans_hcp.predict(X_tissier)
result["silhouette_kmeans_latent_space_tissier"] = str(metrics.silhouette_score(X_tissier, clusters_tissier))

X_tissier_tsne = TSNE(n_components=2, perplexity=8).fit_transform(X_tissier)
clusters_tsne_tissier = kmeans_tsne_hcp.predict(X_tissier_tsne)
result["silhouette_kmeans_tsne_space_tissier"] = str(metrics.silhouette_score(X_tissier_tsne, clusters_tsne_tissier))

####### Accuracy score + ARI score
labels_set = pd.read_csv('/neurospin/dico/data/deep_folding/datasets/ACC_patterns/tissier_labels.csv')
dico_labels = {labels_set['Sujet'][k]: 0 if labels_set['ACC_right_2levels_new'][k]=='absent' else 1 for k in range(len(labels_set))}
labels_sub = [sub for sub in dico_labels.keys()]
labels_cing = [dico_labels[sub] for sub in labels_sub if sub in dico_labels.keys()]

result["accuracy_kmeans_tsne_space_tissier"] = metrics.accuracy_score(clusters_tsne_tissier, labels_cing)
result["ari_kmeans_tsne_space_tissier"] = metrics.adjusted_rand_score(clusters_tsne_tissier, labels_cing)

result["accuracy_kmeans_latent_space_tissier"] = metrics.accuracy_score(clusters_tissier, labels_cing)
result["ari_kmeans_latent_space_tissier"] = metrics.adjusted_rand_score(clusters_tissier, labels_cing)

#### visualization of test_set
df_test = pd.DataFrame()
df_test['latent'] = encoded['test']
X_test = np.array(list(df_test['latent']))

X_test_tsne = TSNE(n_components=2, perplexity=8).fit_transform(X_test)

x = [X_test_tsne[k][0] for k in range(len(X_test_tsne))]
y = [X_test_tsne[k][1] for k in range(len(X_test_tsne))]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x, y)

plt.savefig(root_dir+'test_tsne.png')


#### visualization of tissier
color_dict = {0: 'blue', 1: 'magenta'}

"""============ TSNE on Tissier dataset =============="""
arr = X_tissier_tsne
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)

for g in [0, 1]:
    ix = np.where(np.array(labels_cing) == g)
    x = [arr[ix][k][0] for k in range(len(ix[0]))]
    y = [arr[ix][k][1] for k in range(len(ix[0]))]
    if g==0:
        g_lab = 'no paracingular'
    else:
        g_lab = 'paracingular'
    ax.scatter(x, y, c = color_dict[g], label = g_lab, s=100)

for i, txt in enumerate(labels_sub):
    ax.annotate(txt, (arr[i][0], arr[i][1]))

plt.axis('off')
ax.legend(fontsize=12)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

plt.savefig(f"{root_dir}tissier_tsne_per8.png")


arr = TSNE(n_components=2, perplexity=15).fit_transform(X_tissier)
fig = plt.figure(figsize=(8,8))
ax = fig.add_subplot(111)

for g in [0, 1]:
    ix = np.where(np.array(labels_cing) == g)
    x = [arr[ix][k][0] for k in range(len(ix[0]))]
    y = [arr[ix][k][1] for k in range(len(ix[0]))]
    if g==0:
        g_lab = 'no paracingular'
    else:
        g_lab = 'paracingular'
    ax.scatter(x, y, c = color_dict[g], label = g_lab, s=100)

for i, txt in enumerate(labels_sub):
    ax.annotate(txt, (arr[i][0], arr[i][1]))

plt.axis('off')
ax.legend(fontsize=12)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

plt.savefig(f"{root_dir}tissier_tsne_per15.png")


with open(f"{root_dir}results.json", "w") as json_file:
    print(result)
    json_file.write(json.dumps(result, sort_keys=True, indent=4))
