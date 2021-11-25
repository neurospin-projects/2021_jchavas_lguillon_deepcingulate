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
import json

"""============== Parameters ============="""
n = 20
kl = 1
side='R'

data_dir = '/neurospin/dico/data/deep_folding/current/crops/CINGULATE/mask/sulcus_based/2mm/centered_combined/'
root_dir = f"/neurospin/dico/lguillon/midl_22/gridsearch/n_{n}_kl_{kl}/"
subject_dir = "/neurospin/dico/data/deep_folding/current/"

result = {"test_silhouette_kmeans_latent_space":0, "test_silhouette_kmeans_tsne_space":0,
           "train_silhouette_kmeans_latent_space":0, "train_silhouette_kmeans_tsne_space":0,
           "tissier_silhouette_kmeans_latent_space":0, "tissier_silhouette_kmeans_tsne_space":0,
           "tissier_silhouette_kmeans_fit_tissier_latent_space":0,
           "tissier_silhouette_kmeans_fit_tissier_tsne": 0,
           "tissier_accuracy_kmeans_fit_tissier_tsne_space":0,
           "tissier_accuracy_kmeans_fit_tissier_latent_space":0,
           "tissier_accuracy_kmeans_tsne_space":0,
           "tissier_accuracy_kmeans_latent_space":0
           }

"""=================== Loading of the model ================"""
model = VAE((1, 20, 40, 40), n, depth=3)
model.load_state_dict(torch.load(root_dir + 'checkpoint.pt'))


"""=================== Loading of data ================"""
###### HCP Train + val sets
train_list = pd.read_csv(os.path.join(subject_dir, 'train.csv'), header=None,
                        usecols=[0], names=['subjects'])

train_list['subjects'] = train_list['subjects'].astype('str')

tmp = pd.read_pickle(os.path.join(data_dir, "hcp", f"{side}skeleton.pkl")).T
tmp.index.astype('str')
tmp = tmp.merge(train_list, left_on = tmp.index, right_on='subjects', how='right')
filenames = list(train_list['subjects'])
train_set = datasets.SkeletonDataset(dataframe=tmp, filenames=filenames)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=1,
                                          shuffle=False, num_workers=0)

###### HCP test sets
test_list = pd.read_csv(os.path.join(subject_dir, 'test.csv'), header=None,
                        usecols=[0], names=['subjects'])

test_list['subjects'] = test_list['subjects'].astype('str')

tmp = pd.read_pickle(os.path.join(data_dir, f"{side}skeleton.pkl")).T
tmp.index.astype('str')
tmp = tmp.merge(test_list, left_on = tmp.index, right_on='subjects', how='right')
filenames = list(test_list['subjects'])
test_set = datasets.SkeletonDataset(dataframe=tmp, filenames=filenames)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                          shuffle=False, num_workers=0)

###### Tissier database
labels_set = pd.read_csv('/neurospin/dico/data/deep_folding/datasets/ACC_patterns/tissier_labels.csv')
dico_labels = {labels_set['Sujet'][k]: 0 if labels_set['ACC_right_2levels_new'][k]=='absent' else 1 for k in range(len(labels_set))}
df_tissier_labels = pd.DataFrame(data={'sub': list(dico_labels.keys()), 'cing':list(dico_labels.values())})

tmp2 = pd.read_pickle(os.path.join(data_dir, "tissier_2018", f"{side}skeleton.pkl")).T
tmp2.index.astype('str')

# keeping only labeled subjects
tmp2 = tmp2.merge(df_tissier_labels, left_on=tmp2.index, right_on="sub", how="right")

filenames2 = list(tmp2['sub'])
tissier = datasets.SkeletonDataset(dataframe=tmp2, filenames=filenames2)

labels = np.array(['hcp' for k in range(len(test_list))] \
         + ['tissier' for k in range(len(tmp2))])

tissier = torch.utils.data.DataLoader(tissier, batch_size=1,
                                          shuffle=False, num_workers=0)

dico_set_loaders = {'train': train_loader, 'test': test_loader, 'tissier': tissier}


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


"""=================== Test data: silhouette score on 2 clusters ================"""
####### TSNE ON TEST SET
df_encoded = pd.DataFrame()
df_encoded['latent'] = encoded['test']
X = np.array(list(df_encoded['latent']))

X_embedded = TSNE(n_components=2).fit_transform(X)

x = [X_embedded[k][0] for k in range(len(X))]
y = [X_embedded[k][1] for k in range(len(X))]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x, y)

plt.savefig(root_dir+'test_set_tsne.png')

####### Kmeans ON TEST SET on LATENT SPACE AND REDUCED LATENT SPACE
kmeans2d_train = KMeans(n_clusters=2, random_state=0).fit(X_embedded)
res = kmeans2d_train.predict(X_embedded)
print('on tsne space: ', metrics.silhouette_score(X_embedded, res))
result["test_silhouette_kmeans_tsne_space"] = str(metrics.silhouette_score(X_embedded, res))

kmeans_train = KMeans(n_clusters=2, random_state=0).fit(X)
res = kmeans_train.predict(X)
print('on latent space: ',metrics.silhouette_score(X, res))
result["test_silhouette_kmeans_latent_space"] = str(metrics.silhouette_score(X, res))

####### ACCURACY ON TEST SET
df_encoded = pd.DataFrame()
df_encoded['latent'] = encoded['tissier']
X = np.array(list(df_encoded['latent']))

labels_sub = [sub for sub in dico_labels.keys()]
labels_cing = [dico_labels[sub] for sub in labels_sub if sub in dico_labels.keys()]

X_embedded = TSNE(n_components=2).fit_transform(X)
color_dict = {0: 'blue', 1: 'magenta'}
arr = X_embedded

"""============ TSNE on Tissier dataset =============="""
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

for i, txt in enumerate(filenames2):
    ax.annotate(txt, (arr[i][0], arr[i][1]))

plt.axis('off')
ax.legend(fontsize=12)
ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)

plt.savefig(f"{root_dir}tissier_tsne.png")

"""============ silhouette score on Tissier dataset =============="""
res = kmeans_train.predict(X)
print('on latent space for Tissier: ',metrics.silhouette_score(X, res))
result["tissier_silhouette_kmeans_latent_space"] = str(metrics.silhouette_score(X, res))
result["tissier_accuracy_kmeans_latent_space"] = metrics.accuracy_score(res, labels_cing)

res = kmeans2d_train.predict(X_embedded)
print('on latent space for Tissier: ', metrics.silhouette_score(X_embedded, res))
result["tissier_silhouette_kmeans_tsne_space"] = str(metrics.silhouette_score(X_embedded, res))
result["tissier_accuracy_kmeans_tsne_space"] = metrics.accuracy_score(res, labels_cing)

res_tissier = KMeans(n_clusters=2, random_state=0).fit_predict(X)
print('fit on tissier, on latent space for Tissier: ',metrics.silhouette_score(X, res_tissier))
result["tissier_silhouette_kmeans_fit_tissier_latent_space"] = str(metrics.silhouette_score(X, res_tissier))
result["tissier_accuracy_kmeans_fit_tissier_latent_space"] = metrics.accuracy_score(res_tissier, labels_cing)

res_tissier_embedded = KMeans(n_clusters=2, random_state=0).fit_predict(X_embedded)
print('on latent space for Tissier: ', metrics.silhouette_score(X_embedded, res_tissier_embedded))
result["tissier_silhouette_kmeans_fit_tissier_tsne"] = str(metrics.silhouette_score(X_embedded, res_tissier_embedded))
result["tissier_accuracy_kmeans_fit_tissier_tsne_space"] = metrics.accuracy_score(res_tissier_embedded, labels_cing)


with open(f"{root_dir}results.json", "w") as json_file:
    print(result)
    json_file.write(json.dumps(result, sort_keys=True, indent=4))
