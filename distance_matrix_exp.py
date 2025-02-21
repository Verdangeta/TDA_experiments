## Creates pointclouds and vectorizations of persistence diagrams from the ModelNet dataset.
import sys
import pickle
import neptune

#import pandas as pd
from copy import deepcopy
import numpy as np
import gudhi as gd
import argparse
import os, random
import warnings
from pathlib import Path
import matplotlib.pyplot as plt
import time
from tqdm import tqdm
import trimesh as trm
import glob


import tensorflow as tf
import tensorflow.keras as keras

from utils import DenseRagged, PermopRagged
from utils import sep_dist, measure_dist
from gudhi.representations import DiagramSelector
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import train_test_split

from tensorflow.keras.regularizers import l1, l2
import mtd
import numpy as np
from scipy.stats import entropy

from tqdm import tqdm

import matplotlib.pyplot as plt
from matplotlib import gridspec
# import torch

import seaborn as sns

from sklearn.decomposition import PCA

def train_model(model, train_data, val_data, test_data, PI_train, PI_val, PI_test, callback, name, epochs = 200, logger = None):
    start_time = time.time()
    
    history = model.fit(train_data,
                            PI_train, epochs=epochs,
                            validation_data=(val_data, PI_val),
                            callbacks=[callback], verbose=0)
    
    
    train_PI_prediction = model.predict(train_data)
    val_PI_prediction = model.predict(val_data)
    test_PI_prediction = model.predict(test_data)
    
    # plt.plot(np.array(history.history["loss"]))
    # plt.plot(np.array(history.history["val_loss"]))
    # plt.legend(["loss", "val_loss"])
    # plt.title("Second type model with PCA_dist")
    # plt.show()
    
    val_loss = measure_dist(PI_train, train_PI_prediction, method = "KL_sym")
    train_loss = measure_dist(PI_val, val_PI_prediction, method = "KL_sym")
    test_loss = measure_dist(PI_test, test_PI_prediction, method = "KL_sym")

    logger["train/loss in the end"] = train_loss
    logger["val/loss in the end"] = val_loss
    logger["test/loss in the end"] = test_loss
    logger["fit time"] = time.time() - start_time
    logger["train/loss"] = history["loss"]
    logger["val/loss"] = history["val_loss"]
    
    
    print(f"Last train loss: {train_loss:.2f} and val loss: {val_loss:.2f} and test loss {test_loss:.2f}")
    print("Model fitted for --- %s seconds ---" % (time.time() - start_time))
    return history


def create_model_base(cloud_dim, patience = 100):
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-8, 
                                            patience=patience, verbose=0, mode='auto', baseline=None, 
                                            restore_best_weights=True)
    
    dropout_rate = 0.2
    optim = tf.keras.optimizers.Adamax(learning_rate=1e-4)
    inputs_1 = tf.keras.Input(shape=(None, cloud_dim), dtype ="float32", ragged=True)
    inputs_2 = tf.keras.Input(shape=(None, cloud_dim), dtype ="float32", ragged=True)
    # inputs_dist = tf.keras.Input(shape=(N_points, dist_dim), dtype ="float32", ragged=False)
    
    x = DenseRagged(units=30, use_bias=True, activation='relu')(inputs_1)
    x = DenseRagged(units=20, use_bias=True, activation='relu')(x)
    x = DenseRagged(units=10, use_bias=True, activation='relu')(x)
    x = PermopRagged()(x)
    
    y = DenseRagged(units=30, use_bias=True, activation='relu')(inputs_2)
    y = DenseRagged(units=20, use_bias=True, activation='relu')(y)
    y = DenseRagged(units=10, use_bias=True, activation='relu')(y)
    y = PermopRagged()(y)
    
    x_y = keras.layers.Concatenate(axis=1)([inputs_1, inputs_2])
    x_y = DenseRagged(units=30, use_bias=True, activation='relu')(x_y)
    x_y = DenseRagged(units=20, use_bias=True, activation='relu')(x_y)
    x_y = DenseRagged(units=10, use_bias=True, activation='relu')(x_y)
    x_y= PermopRagged()(x_y)
    
    # d = DenseRagged(units=30, use_bias=True, activation='relu')(inputs_dist)
    # d = DenseRagged(units=20, use_bias=True, activation='relu')(d)
    # d = DenseRagged(units=10, use_bias=True, activation='relu')(d)
    # d = PermopRagged()(d)
    
    
    z = keras.layers.Concatenate(axis=-1)([x, y, x_y])
    
    # z = keras.layers.Concatenate(axis=-1)([x, y, x_y])
    z = tf.keras.layers.Normalization()(z)
    z = tf.keras.layers.Dense(150,  activation='relu', kernel_regularizer=l2(1e-4), activity_regularizer=l2(1e-4))(z)
    z = tf.keras.layers.Dropout(dropout_rate)(z)
    z = tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=l2(1e-4), activity_regularizer=l2(1e-4))(z)
    z = tf.keras.layers.Dropout(dropout_rate)(z)
    z = tf.keras.layers.Dense(400, activation='relu', kernel_regularizer=l2(1e-3), activity_regularizer=l2(1e-4))(z)
    z = tf.keras.layers.Dropout(dropout_rate)(z)
    outputs = tf.keras.layers.Dense(PI_train.shape[1], activation='sigmoid')(z)
    outputs = tf.keras.layers.Lambda(lambda x: tf.experimental.numpy.clip(x, 1e-8, None))(outputs) ## for preventiong calculating log(0)
    outputs = tf.keras.layers.Lambda(lambda x: x / (tf.reduce_sum(x, axis=-1, keepdims=True) + 1e-7))(outputs)
    model_PI_second_type = tf.keras.Model(inputs=[inputs_1, inputs_2], outputs=outputs)
    model_PI_second_type.compile(optimizer=optim, loss=sym_KL)
    return model_PI_second_type, callback


def create_model_with_distance_matrix(cloud_dim, dist_dim, patience = 100):
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-8, 
                                            patience=patience, verbose=0, mode='auto', baseline=None, 
                                            restore_best_weights=True)
    
    dropout_rate = 0.2
    optim = tf.keras.optimizers.Adamax(learning_rate=1e-4)
    inputs_1 = tf.keras.Input(shape=(None, cloud_dim), dtype ="float32", ragged=True)
    inputs_2 = tf.keras.Input(shape=(None, cloud_dim), dtype ="float32", ragged=True)
    inputs_dist = tf.keras.Input(shape=(None, dist_dim), dtype ="float32", ragged=False)
    
    x = DenseRagged(units=30, use_bias=True, activation='relu')(inputs_1)
    x = DenseRagged(units=20, use_bias=True, activation='relu')(x)
    x = DenseRagged(units=10, use_bias=True, activation='relu')(x)
    x = PermopRagged()(x)
    
    y = DenseRagged(units=30, use_bias=True, activation='relu')(inputs_2)
    y = DenseRagged(units=20, use_bias=True, activation='relu')(y)
    y = DenseRagged(units=10, use_bias=True, activation='relu')(y)
    y = PermopRagged()(y)
    
    x_y = keras.layers.Concatenate(axis=1)([inputs_1, inputs_2])
    x_y = DenseRagged(units=30, use_bias=True, activation='relu')(x_y)
    x_y = DenseRagged(units=20, use_bias=True, activation='relu')(x_y)
    x_y = DenseRagged(units=10, use_bias=True, activation='relu')(x_y)
    x_y= PermopRagged()(x_y)
    
    d = DenseRagged(units=30, use_bias=True, activation='relu')(inputs_dist)
    d = DenseRagged(units=20, use_bias=True, activation='relu')(d)
    d = DenseRagged(units=10, use_bias=True, activation='relu')(d)
    d = PermopRagged()(d)
    
    
    z = keras.layers.Concatenate(axis=-1)([x, y, x_y, d])
    
    # z = keras.layers.Concatenate(axis=-1)([x, y, x_y])
    z = tf.keras.layers.Normalization()(z)
    z = tf.keras.layers.Dense(150,  activation='relu', kernel_regularizer=l2(1e-4), activity_regularizer=l2(1e-4))(z)
    z = tf.keras.layers.Dropout(dropout_rate)(z)
    z = tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=l2(1e-4), activity_regularizer=l2(1e-4))(z)
    z = tf.keras.layers.Dropout(dropout_rate)(z)
    z = tf.keras.layers.Dense(400, activation='relu', kernel_regularizer=l2(1e-3), activity_regularizer=l2(1e-4))(z)
    z = tf.keras.layers.Dropout(dropout_rate)(z)
    outputs = tf.keras.layers.Dense(PI_train.shape[1], activation='sigmoid')(z)
    outputs = tf.keras.layers.Lambda(lambda x: tf.experimental.numpy.clip(x, 1e-8, None))(outputs) ## for preventiong calculating log(0)
    outputs = tf.keras.layers.Lambda(lambda x: x / (tf.reduce_sum(x, axis=-1, keepdims=True) + 1e-7))(outputs)
    model_PI_second_type = tf.keras.Model(inputs=[inputs_1, inputs_2, inputs_dist], outputs=outputs)
    model_PI_second_type.compile(optimizer=optim, loss=sym_KL)
    return model_PI_second_type, callback



# sns.set_theme(style="darkgrid")
FIG_SIZE = (18,5)
sns.set_context("talk")
plt.style.use('ggplot')

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


from scipy.stats import entropy

def sym_KL(y_true, y_pred):
    loss = tf.keras.losses.KLDivergence()
    return (loss(y_true, y_pred) + loss(y_pred, y_true))/2

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # Настройки для детерминированного поведения
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'





import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default='Syntethic')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dist_nfeatures", type=int, default=60)
parser.add_argument("--use_pca", action='store_true', default=False)
parser.add_argument("--n_components", type=int, default=10)
parser.add_argument("--run_name", type=str, default="Base")


args = parser.parse_args()

seed = args.seed
run_name = args.run_name
task = args.task
n_features = args.dist_nfeatures



run = neptune.init_run(
    project="verdangeta/CrossRipsNet-distance-matrix",
    api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJjMTA4OWNjMS0xNmNlLTRjMjctYWY2MC1mMmViOGE0Y2ZmYjgifQ==",
)  # your credentials

run["parameters"] = args
# run["run_name"] = run_name
# run["seed"] = seed
# run["task"] = task
# run["dist_features"] = n_features
# run["use_pca"] = args.use_pca



seed_everything(seed=seed)


if task == "3D_shapes":
    N_points = 1024
    num_samples=1024
    cloud_dim = 3
    
    with open('Data/cross_ripsnet_3d_exp/3d_shapes_pc_train_2500_30_boostrap', 'rb') as fp:
        pc_train = pickle.load(fp)

    with open("Data/cross_ripsnet_3d_exp/3d_shapes_indexes_2500_30_boostrap", "rb") as fp:
        train_indexes = pickle.load(fp)
    
    
    with open("Data/cross_ripsnet_3d_exp/3d_shapes_PI_2500_30_boostrap", 'rb') as fp:   #Pickling
        PI_train_all = pickle.load(fp)
    
    # PI_train = np.vstack([PI_train_all[:2200]])
    # clean_PI_test = np.vstack([PI_train_all[2200:]])

    mid = 2200
    end = -1

elif task == "Syntethic":
    cloud_dim = 2
    
    with open('RipsNet_exp/cross_pd_circles_3000_strat_boost_10_data.npy', 'rb') as f:
        pc_train = np.load(f)
    with open('RipsNet_exp/cross_pd_circles_3000_strat_boost_10_indexes.npy', 'rb') as f:
        train_indexes = np.load(f)
    with open("RipsNet_exp/cross_pd_circles_3000_strat_boost_10_PI.npy", 'rb') as fp:   #Pickling
        PI_train_all = pickle.load(fp)


    train_indexes = np.vstack([train_indexes[:800], train_indexes[1000:1800], train_indexes[2000:2800], train_indexes[800:1000], train_indexes[1800:2000], train_indexes[2800:3000]])
    PI_train = np.vstack([PI_train_all[:800], PI_train_all[1000:1800], PI_train_all[2000:2800], PI_train_all[800:1000], PI_train_all[1800:2000], PI_train_all[2800:3000]])
    mid = 2400
    end = -1
        
elif task == "Textual":

    with open("Data/cross_ripsnet_text_exp/human_gpt3_davinci_003_pc_train", "rb") as fp:   #Pickling
        pc_train = pickle.load(fp)
    
    with open("Data/cross_ripsnet_text_exp/human_gpt3_davinci_003_train_indexes", "rb") as fp:   #Pickling
        train_indexes = pickle.load(fp)
    
    
    with open("Data/cross_ripsnet_text_exp/human_gpt3_davinci_003_PI_10000_50_boostrap", 'rb') as fp:   #Pickling
        PI_train_all = pickle.load(fp)

    
    # PI_train = np.vstack([PI_train_all[:2000]])
    # clean_PI_test = np.vstack([PI_train_all[2000:2500]])
    cloud_dim = pc_train.shape[-1]
    mid = 2000
    end = 2500


print("Data downloaded")
    

if args.use_pca:
    n_components = args.n_components
    pca = PCA(n_components=n_components)
    features = [pca.fit_transform(cloud) for cloud in pc_train]
    cloud_dim = n_components
    print(f"Data dimension reduced to {n_components}")
else:
    features = pc_train


# if task in ["Syntethic", "3D_shapes"]:
#     start_time = time.time()
#     pdist_train = np.zeros((data_train_concat.shape[0], N_points, 2*N_points))
#     for i in tqdm(range(data_train_concat.shape[0]), desc = "pdist for train dataset"):
#         cloud_1 = data_train_concat[i, :N_points]
#         cloud_2 = data_train_concat[i, N_points:]
#         # NB !
#         # Swapping point clouds for consistency with the paper
#         cloud_1, cloud_2 = cloud_2, cloud_1
        
#         d = sep_dist(cloud_1, cloud_2, pdist_device = "cuda")
#         m = d[cloud_1.shape[0]:, :cloud_1.shape[0]].mean()
#         d[:cloud_1.shape[0]][:cloud_1.shape[0]] = 0
#         d[d < m*(1e-6)] = 0
#         # pdist_train.append(d)
#         pdist_train[i, :, :] = d[N_points:, :] ## because all upper numbers equal to zero
    
#     pdist_test = np.zeros((data_test_concat.shape[0], N_points, 2*N_points))
#     for i in tqdm(range(data_test_concat.shape[0]), desc = "pdist for test dataset"):
#         cloud_1 = data_test_concat[i, :N_points]
#         cloud_2 = data_test_concat[i, N_points:]
#         # NB !
#         # Swapping point clouds for consistency with the paper
#         cloud_1, cloud_2 = cloud_2, cloud_1
        
#         d = sep_dist(cloud_1, cloud_2, pdist_device = "cuda")
#         m = d[cloud_1.shape[0]:, :cloud_1.shape[0]].mean()
#         d[:cloud_1.shape[0]][:cloud_1.shape[0]] = 0
#         d[d < m*(1e-6)] = 0
#         pdist_test[i, :, :] = d[N_points:, :] ## because all upper numbers equal to zero
        
#     print("Distance maatrix calculationn took --- %s seconds ---" % (time.time() - start_time))

# elif task == "Textual":
start_time = time.time()
pdist_train = []
for idx in tqdm(train_indexes[:mid], desc = "pdist for train dataset"):
    cloud_1 = pc_train[idx[0]]
    cloud_2 = pc_train[idx[1]]
    # NB !
    # Swapping point clouds for consistency with the paper
    cloud_1, cloud_2 = cloud_2, cloud_1
    
    d = sep_dist(cloud_1, cloud_2, pdist_device = "cuda")
    m = d[cloud_1.shape[0]:, :cloud_1.shape[0]].mean()
    d[:cloud_1.shape[0]][:cloud_1.shape[0]] = 0
    d[d < m*(1e-6)] = 0
    # pdist_train.append(d)
    pdist_train.append(d[cloud_1.shape[0]:, :]) ## because all upper numbers equal to zero

pdist_test = []
for idx in tqdm(train_indexes[mid:end], desc = "pdist for test dataset"):
    cloud_1 = pc_train[idx[0]]
    cloud_2 = pc_train[idx[1]]
    # NB !
    # Swapping point clouds for consistency with the paper
    cloud_1, cloud_2 = cloud_2, cloud_1
    
    d = sep_dist(cloud_1, cloud_2, pdist_device = "cuda")
    m = d[cloud_1.shape[0]:, :cloud_1.shape[0]].mean()
    d[:cloud_1.shape[0]][:cloud_1.shape[0]] = 0
    d[d < m*(1e-6)] = 0
    pdist_test.append(d[cloud_1.shape[0]:, :]) ## because all upper numbers equal to zero
    
print("Distance matrix calculation took --- %s seconds ---" % (time.time() - start_time))



PI_train = np.vstack([PI_train_all[:mid]])
PI_test = np.vstack([PI_train_all[mid:end]])

tf_data_train_1 = tf.ragged.constant([
    features[i] for i in train_indexes[:mid][:,0]], ragged_rank=1)
tf_data_train_2 = tf.ragged.constant([
    features[i] for i in train_indexes[:mid][:,1]], ragged_rank=1)


tf_data_test_1 = tf.ragged.constant([
    features[i] for i in train_indexes[mid:end][:,0]], ragged_rank=1)
tf_data_test_2 = tf.ragged.constant([
    features[i] for i in train_indexes[mid:end][:,1]], ragged_rank=1)

print(f"Data inserted in ragged tensors")





start_time = time.time()

# pca = PCA(n_components=n_features)
# pdist_train_reducted_pca = tf.ragged.constant([pca.fit_transform(cloud) for cloud in pdist_train], ragged_rank=1)
# pdist_test_reducted_pca = tf.ragged.constant([pca.fit_transform(cloud) for cloud in pdist_test], ragged_rank=1)

pdist_train_reducted_pca = tf.ragged.constant(np.random.randn(len(pdist_train), 40, n_features), ragged_rank=1)
pdist_test_reducted_pca = tf.ragged.constant(np.random.randn(len(pdist_train), 40, n_features), ragged_rank=1)
                                     

print("Distance matrix reducted by PCA for --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()


# pdist_train_reducted_max = tf.ragged.constant([np.sort(cloud, axis=1)[:,-n_features:] for cloud in pdist_train], ragged_rank=1)
# pdist_test_reducted_max  = tf.ragged.constant([np.sort(cloud, axis=1)[:,-n_features:] for cloud in pdist_test], ragged_rank=1)

pdist_train_reducted_max = tf.ragged.constant(np.random.randn(len(pdist_train), 40, n_features), ragged_rank=1)
pdist_test_reducted_max = tf.ragged.constant(np.random.randn(len(pdist_train), 40, n_features), ragged_rank=1)
                                    
print("Distance matrix reducted by MAX for --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()

# percentiles = np.linspace(0, 100, n_features)

# pdist_train_reducted_quant = tf.ragged.constant(
#     [np.percentile(cloud, percentiles, axis=-1, method = "closest_observation").transpose((1,0)) for cloud in pdist_train],
#     ragged_rank=1)

# pdist_test_reducted_quant = tf.ragged.constant(
#     [np.percentile(cloud, percentiles, axis=-1, method = "closest_observation").transpose((1,0)) for cloud in pdist_test], 
#     ragged_rank=1)

pdist_train_reducted_quant = tf.ragged.constant(np.random.randn(len(pdist_train), 40, n_features), ragged_rank=1)
pdist_test_reducted_quant = tf.ragged.constant(np.random.randn(len(pdist_train), 40, n_features), ragged_rank=1)

print("Distance matrix reducted by Quantile for --- %s seconds ---" % (time.time() - start_time))


# TRAIN MODELS
complects_of_data_train = [[tf_data_train_1, tf_data_train_2, pdist_train_reducted_pca],
                        [tf_data_train_1, tf_data_train_2, pdist_train_reducted_max], 
                        [tf_data_train_1, tf_data_train_2, pdist_train_reducted_quant]]

complects_of_data_test = [[tf_data_test_1, tf_data_test_2, pdist_test_reducted_pca],
                        [tf_data_test_1, tf_data_test_2, pdist_test_reducted_max], 
                        [tf_data_test_1, tf_data_test_2, pdist_test_reducted_quant]]

names = ["PCA", "MAX", "QUANT"]

for train_data, test_data, name in zip(complects_of_data_train, complects_of_data_test, names):

    # Объединяем данные в датасет
    dataset = tf.data.Dataset.from_tensor_slices((*train_data, PI_train))
    
    # Перемешиваем
    dataset = dataset.shuffle(len(PI_train), seed=seed)
    
    # Разбиваем на train и val
    train_size = int(0.9 * len(PI_train))
    train_dataset = dataset.take(train_size)
    val_dataset = dataset.skip(train_size)
    
    # Разворачиваем обратно в списки
    X_train_1, X_train_2, X_train_dist, y_train = zip(*train_dataset)
    X_train_1, X_train_2, X_train_dist = (tf.ragged.constant(a, ragged_rank=1) for a in [X_train_1, X_train_2, X_train_dist])
    y_train = np.array(y_train)
    X_val_1, X_val_2, X_val_dist, y_val = zip(*val_dataset)
    X_val_1, X_val_2, X_val_dist = (tf.ragged.constant(a, ragged_rank=1) for a in [X_val_1, X_val_2, X_val_dist])
    y_val = np.array(y_val)
    
    print(y_train[0])
    print(type(X_train_1), type(X_train_2), type(X_train_dist), type(y_train))
    print(len(X_train_1), len(X_train_2), len(X_train_dist), len(y_train))
    print(X_train_1[0].shape, X_train_2[0].shape, X_train_dist[0].shape, y_train[0].shape)

    # X_train_1, X_val_1, X_train_2, X_val_2, X_train_dist, X_val_dist, y_train, y_val = train_test_split(*train_data,
    #                                                                           PI_train, test_size=0.1,
    #                                                                           random_state=seed)
    
    model, callback = create_model_with_distance_matrix(cloud_dim, n_features, patience = 100)
    history = train_model(model, [X_train_1, X_train_2, X_train_dist],
                          [X_val_1, X_val_2, X_val_dist], test_data, y_train, y_val, PI_test,
                          callback, name, epochs = 200, logger = run)

model, callback = create_model_base(cloud_dim, patience = 100)
train_data = [tf_data_train_1, tf_data_train_2]
X_train_1, X_val_1, X_train_2, X_val_2, y_train, y_val = train_test_split(*train_data,
                                                                              PI_train, test_size=0.1,
                                                                              random_state=seed)
test_data = [tf_data_test_1, tf_data_test_2]
history = train_model(model, [X_train_1, X_train_2],
                          [X_val_1, X_val_2], test_data, y_train, y_val, PI_test,
                          callback, name, epochs = 200, logger = run)

run.stop()





