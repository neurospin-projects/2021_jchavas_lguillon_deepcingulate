# -*- coding: utf-8 -*-
# /usr/bin/env python3

# Imports
import torch
import pandas as pd
#from preprocessing import create_aims_set
from vae import *
from sklearn.metrics import classification_report
from sklearn import metrics
import datasets

side='R'
data_dir = '/neurospin/dico/data/deep_folding/current/crops/CINGULATE/mask/sulcus_based/2mm/'

root_dir ='/neurospin/dico/lguillon/midl_22/run_2/'
model = VAE((1, 12, 48, 48), 75, depth=3)
model.load_state_dict(torch.load(root_dir + 'checkpoint.pt'))

# fetch data
test_csv = '/neurospin/dico/data/deep_folding/current/test.csv'
test_list = pd.read_csv(test_csv, header=None, usecols=[0], names=['subjects'])
test_list['subjects'] = test_list['subjects'].astype('str')

# encode data
tmp = pd.read_pickle(os.path.join(data_dir, f"{side}skeleton.pkl")).T
#tmp = tmp.rename(columns={0:'subjects'})
tmp.index.astype('str')

tmp = tmp.merge(test_list, left_on = tmp.index, right_on='subjects', how='right')
#tmp.merge(test_list, left_on = tmp.index, right_on='subjects', how='right')
filenames = list(test_list.subjects)
test_set = datasets.SkeletonDataset(dataframe=tmp, filenames=filenames)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=1,
                                          shuffle=True, num_workers=0)

dico_set_loaders = {'test': test_loader}

device = torch.device("cuda", index=0)
model = model.to(device)

class_weights = torch.FloatTensor([1, 200, 27, 356]).to(device)
#class_weights = torch.FloatTensor([2, 7])
distance = nn.CrossEntropyLoss(weight=class_weights, reduction='sum')

# HCP -------- dataset_test_loader & benchmark
tester = ModelTester(model=model, dico_set_loaders=dico_set_loaders,
                     loss_func=distance, kl_weight=2, n_latent=125,
                     depth=3, skeleton=True, root_dir=root_dir)

results = tester.test()

losses = {loader_name:[results[loader_name][k][0] for k in results[loader_name].keys()] for loader_name in dico_set_loaders.keys()}
encoded = {loader_name:[results[loader_name][k][3] for k in results[loader_name].keys()] for loader_name in dico_set_loaders.keys()}

df_encoded = pd.DataFrame()

df_encoded['latent'] = encoded['test']

X = np.array(list(df_encoded['latent']))

from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2).fit_transform(X)

X_embedded = TSNE(n_components=2).fit_transform(X)
color_dict = {'test': 'blue', 'benchmark': 'magenta'}

x = [X_embedded[k][0] for k in range(150)]
y = [X_embedded[k][1] for k in range(150)]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.scatter(x, y)

plt.savefig(root_dir+'tsne.png')

# Kmeans
from sklearn.cluster import KMeans
kmeans2d = KMeans(n_clusters=2, random_state=0).fit_predict(X_embedded)
print('on tsne space: ', metrics.silhouette_score(X_embedded, kmeans2d))

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0).fit_predict(X)

print('on latent space: ',metrics.silhouette_score(X, kmeans))
