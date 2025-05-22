## Creates pointclouds and vectorizations of persistence diagrams from the ModelNet dataset.
import pickle
import neptune

#import pandas as pd
from copy import deepcopy
import numpy as np
import argparse
import os, random

import matplotlib.pyplot as plt
import time
from tqdm import tqdm
from sklearn.model_selection import train_test_split



import tensorflow as tf
import tensorflow.keras as keras

from utils import DenseRagged, PermopRagged
from utils import sep_dist, measure_dist


from tensorflow.keras.regularizers import l2
import numpy as np

from tqdm import tqdm

import matplotlib.pyplot as plt
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
    
    
    # print(PI_train.shape, train_PI_prediction.shape)
    val_loss = np.mean(measure_dist(PI_val, val_PI_prediction, method = "KL_sym"))
    train_loss =  np.mean(measure_dist(PI_train, train_PI_prediction, method = "KL_sym"))
    test_loss =  np.mean(measure_dist(PI_test, test_PI_prediction, method = "KL_sym"))

    if logger:
        logger[f"{name}/train/loss end"] = train_loss
        logger[f"{name}/val/loss end"] = val_loss
        logger[f"{name}/test/loss end"] = test_loss
        logger[f"{name}/fit_time"] = time.time() - start_time

        for value in history.history["loss"][20:]:
            logger[f"{name}/train/loss"].append(value)
        for value in history.history["val_loss"][20:]:
            logger[f"{name}/val/loss"].append(value)
    
    
    print(f"Last train loss {name}: {train_loss:.2f},  val loss: {val_loss:.2f} and test loss {test_loss:.2f}")
    print("Model fitted for --- %s seconds ---" % (time.time() - start_time))
    return history


def create_model_RipsNet(cloud_dim, patience = 100):
    callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=1e-8, 
                                            patience=patience, verbose=0, mode='auto', baseline=None, 
                                            restore_best_weights=True)
    
    optim = tf.keras.optimizers.Adamax(learning_rate=1e-4)
    inputs_1 = tf.keras.Input(shape=(None, cloud_dim), dtype ="float32", ragged=True)
    inputs_2 = tf.keras.Input(shape=(None, cloud_dim), dtype ="float32", ragged=True)
    
    x_y = keras.layers.Concatenate(axis=1)([inputs_1, inputs_2])
    x_y = DenseRagged(units=30, use_bias=True, activation='relu')(x_y)
    x_y = DenseRagged(units=20, use_bias=True, activation='relu')(x_y)
    x_y = DenseRagged(units=10, use_bias=True, activation='relu')(x_y)
    x_y= PermopRagged()(x_y)
    

    z = tf.keras.layers.Dense(150,  activation='relu', kernel_regularizer=l2(1e-4), activity_regularizer=l2(1e-4))(x_y)
    z = tf.keras.layers.Dense(200, activation='relu', kernel_regularizer=l2(1e-4), activity_regularizer=l2(1e-4))(z)
    z = tf.keras.layers.Dense(400, activation='relu', kernel_regularizer=l2(1e-3), activity_regularizer=l2(1e-4))(z)
    outputs = tf.keras.layers.Dense(PI_train.shape[1], activation='sigmoid')(z)
    outputs = tf.keras.layers.Lambda(lambda x: tf.experimental.numpy.clip(x, 1e-8, None))(outputs) ## for preventiong calculating log(0)
    outputs = tf.keras.layers.Lambda(lambda x: x / (tf.reduce_sum(x, axis=-1, keepdims=True) + 1e-7))(outputs)
    model_PI_second_type = tf.keras.Model(inputs=[inputs_1, inputs_2], outputs=outputs)
    model_PI_second_type.compile(optimizer=optim, loss=sym_KL)
    return model_PI_second_type, callback


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
    
    # y = DenseRagged(units=30, use_bias=True, activation='relu')(inputs_2)
    # y = DenseRagged(units=20, use_bias=True, activation='relu')(y)
    # y = DenseRagged(units=10, use_bias=True, activation='relu')(y)
    # y = PermopRagged()(y)
    
    x_y = keras.layers.Concatenate(axis=1)([inputs_1, inputs_2])
    x_y = DenseRagged(units=30, use_bias=True, activation='relu')(x_y)
    x_y = DenseRagged(units=20, use_bias=True, activation='relu')(x_y)
    x_y = DenseRagged(units=10, use_bias=True, activation='relu')(x_y)
    x_y= PermopRagged()(x_y)
    
    # d = DenseRagged(units=30, use_bias=True, activation='relu')(inputs_dist)
    # d = DenseRagged(units=20, use_bias=True, activation='relu')(d)
    # d = DenseRagged(units=10, use_bias=True, activation='relu')(d)
    # d = PermopRagged()(d)
    
    
    # z = keras.layers.Concatenate(axis=-1)([x, y, x_y])
    z = keras.layers.Concatenate(axis=-1)([x, x_y])
    
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
    inputs_dist = tf.keras.Input(shape=(None, dist_dim), dtype ="float32", ragged=True)
    
    x = DenseRagged(units=30, use_bias=True, activation='relu')(inputs_1)
    x = DenseRagged(units=20, use_bias=True, activation='relu')(x)
    x = DenseRagged(units=10, use_bias=True, activation='relu')(x)
    x = PermopRagged()(x)
    
    # y = DenseRagged(units=30, use_bias=True, activation='relu')(inputs_2)
    # y = DenseRagged(units=20, use_bias=True, activation='relu')(y)
    # y = DenseRagged(units=10, use_bias=True, activation='relu')(y)
    # y = PermopRagged()(y)
    
    x_y = keras.layers.Concatenate(axis=1)([inputs_1, inputs_2])
    x_y = DenseRagged(units=30, use_bias=True, activation='relu')(x_y)
    x_y = DenseRagged(units=20, use_bias=True, activation='relu')(x_y)
    x_y = DenseRagged(units=10, use_bias=True, activation='relu')(x_y)
    x_y= PermopRagged()(x_y)
    
    d = DenseRagged(units=30, use_bias=True, activation='relu')(inputs_dist)
    d = DenseRagged(units=20, use_bias=True, activation='relu')(d)
    d = DenseRagged(units=10, use_bias=True, activation='relu')(d)
    d = PermopRagged()(d)
    
    
    # z = keras.layers.Concatenate(axis=-1)([x, y, x_y, d])
    z = keras.layers.Concatenate(axis=-1)([x, x_y, d])

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


def sym_KL(y_true, y_pred):
    loss = tf.keras.losses.KLDivergence()
    return (loss(y_true, y_pred) + loss(y_pred, y_true))/2

def seed_everything(seed=42):
    random.seed(seed)
    # os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    
    # # Настройки для детерминированного поведения
    # os.environ['TF_DETERMINISTIC_OPS'] = '1'
    # os.environ['TF_CUDNN_DETERMINISTIC'] = '1'





import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default='Syntethic')
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--dist_nfeatures", type=int, default=60)
parser.add_argument("--use_pca", action='store_true', default=False)
parser.add_argument("--n_components", type=int, default=10)
parser.add_argument("--epochs", type=int, default=3000)
parser.add_argument("--run_name", type=str, default="Base")
parser.add_argument("--patience", type=int, default=100)


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



seed_everything(seed=seed)


if task == "3D_shapes":
    N_points = 1024
    num_samples = 1024
    cloud_dim = 3
    
    with open('Data/cross_ripsnet_3d_exp/3d_shapes_pc_train_2500_30_boostrap', 'rb') as fp:
        pc_train = pickle.load(fp)

    with open("Data/cross_ripsnet_3d_exp/3d_shapes_indexes_2500_30_boostrap", "rb") as fp:
        train_indexes = pickle.load(fp)
    
    
    with open("Data/cross_ripsnet_3d_exp/3d_shapes_PI_2500_30_boostrap", 'rb') as fp:   #Pickling
        PI_all = pickle.load(fp)
    
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
        PI_all = pickle.load(fp)


    train_indexes = np.vstack([train_indexes[:800], train_indexes[1000:1800], train_indexes[2000:2800], train_indexes[800:1000], train_indexes[1800:2000], train_indexes[2800:3000]])
    PI_all = np.vstack([PI_all[:800], PI_all[1000:1800], PI_all[2000:2800], PI_all[800:1000], PI_all[1800:2000], PI_all[2800:3000]])
    mid = 2400
    end = -1
        
elif task == "Textual":

    with open("Data/cross_ripsnet_text_exp/human_gpt3_davinci_003_pc_train", "rb") as fp:   #Pickling
        pc_train = pickle.load(fp)
    
    with open("Data/cross_ripsnet_text_exp/human_gpt3_davinci_003_train_indexes", "rb") as fp:   #Pickling
        train_indexes = pickle.load(fp)
    
    
    with open("Data/cross_ripsnet_text_exp/human_gpt3_davinci_003_PI_10000_50_boostrap", 'rb') as fp:   #Pickling
        PI_all = pickle.load(fp)

    
    cloud_dim = pc_train[0].shape[-1]
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



PI_train = np.vstack([PI_all[:mid]])
PI_test = np.vstack([PI_all[mid:end]])

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

pca = PCA(n_components=n_features)
pdist_train_reducted_pca = tf.ragged.constant([pca.fit_transform(cloud) for cloud in pdist_train], ragged_rank=1)
pdist_test_reducted_pca = tf.ragged.constant([pca.fit_transform(cloud) for cloud in pdist_test], ragged_rank=1)

# pdist_train_reducted_pca = tf.ragged.constant(np.random.randn(len(pdist_train), 40, n_features), ragged_rank=1)
# pdist_test_reducted_pca = tf.ragged.constant(np.random.randn(len(pdist_test), 40, n_features), ragged_rank=1)
                                     

print("Distance matrix reducted by PCA for --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()


pdist_train_reducted_max = tf.ragged.constant([np.sort(cloud, axis=1)[:,-n_features:] for cloud in pdist_train], ragged_rank=1)
pdist_test_reducted_max  = tf.ragged.constant([np.sort(cloud, axis=1)[:,-n_features:] for cloud in pdist_test], ragged_rank=1)

# pdist_train_reducted_max = tf.ragged.constant(np.random.randn(len(pdist_train), 40, n_features), ragged_rank=1)
# pdist_test_reducted_max = tf.ragged.constant(np.random.randn(len(pdist_test), 40, n_features), ragged_rank=1)
                                    
print("Distance matrix reducted by MAX for --- %s seconds ---" % (time.time() - start_time))

start_time = time.time()

percentiles = np.linspace(0, 100, n_features)

pdist_train_reducted_quant = tf.ragged.constant(
    [np.percentile(cloud, percentiles, axis=-1, interpolation = "nearest").transpose((1,0)) for cloud in pdist_train],
    ragged_rank=1)

pdist_test_reducted_quant = tf.ragged.constant(
    [np.percentile(cloud, percentiles, axis=-1, interpolation = "nearest").transpose((1,0)) for cloud in pdist_test], 
    ragged_rank=1)

# pdist_train_reducted_quant = tf.ragged.constant(np.random.randn(len(pdist_train), 40, n_features), ragged_rank=1)
# pdist_test_reducted_quant = tf.ragged.constant(np.random.randn(len(pdist_test), 40, n_features), ragged_rank=1)

print("Distance matrix reducted by Quantile for --- %s seconds ---" % (time.time() - start_time))


# TRAIN MODELS
complects_of_data_train = [[tf_data_train_1, tf_data_train_2, pdist_train_reducted_pca],
                        [tf_data_train_1, tf_data_train_2, pdist_train_reducted_max], 
                        [tf_data_train_1, tf_data_train_2, pdist_train_reducted_quant]]

complects_of_data_test = [[tf_data_test_1, tf_data_test_2, pdist_test_reducted_pca],
                        [tf_data_test_1, tf_data_test_2, pdist_test_reducted_max], 
                        [tf_data_test_1, tf_data_test_2, pdist_test_reducted_quant]]

names = ["PCA", "MAX", "QUANT"]

train_idxs, val_idxs = train_test_split(
    list(range(len(PI_train))), test_size=0.1, random_state=seed)


bar = tqdm(total=len(names) + 2)
for train_data, test_data, name in zip(complects_of_data_train, complects_of_data_test, names):
    bar.set_description(f"Training {name} model!")

    X_train_1, X_train_2, X_train_dist = [tf.gather(data, train_idxs, axis=0) for data in train_data]
    y_train = np.take(PI_train, train_idxs, axis=0)
    X_val_1, X_val_2, X_val_dist = [tf.gather(data, val_idxs, axis=0) for data in train_data]
    y_val = np.take(PI_train, val_idxs, axis = 0)


    model, callback = create_model_with_distance_matrix(cloud_dim, n_features, patience = args.patience)
    history = train_model(model, [X_train_1, X_train_2, X_train_dist],
                          [X_val_1, X_val_2, X_val_dist], test_data, y_train, y_val, PI_test,
                          callback, name, epochs = args.epochs, logger = run)
    bar.update()

# Run Cross-RipsNet model without a distance matrix
bar.set_description(f"Training model without distance matrix!")
model, callback = create_model_base(cloud_dim, patience = args.patience)

train_data = [tf_data_train_1, tf_data_train_2]

X_train_1, X_train_2  = [tf.gather(data, train_idxs, axis=0) for data in train_data]
y_train = np.take(PI_train, train_idxs, axis=0)
X_val_1, X_val_2 = [tf.gather(data, val_idxs, axis=0) for data in train_data]
y_val = np.take(PI_train, val_idxs, axis = 0)


test_data = [tf_data_test_1, tf_data_test_2]

name = "BASE"
history = train_model(model, [X_train_1, X_train_2],
                          [X_val_1, X_val_2], test_data, y_train, y_val, PI_test,
                          callback, name, epochs = args.epochs, logger = run)
bar.update()

# Run RipsNet model without a distance matrix
bar.set_description(f"Training RipsNet model without distance matrix!")
model, callback = create_model_RipsNet(cloud_dim, patience = args.patience)

train_data = [tf_data_train_1, tf_data_train_2]

X_train_1, X_train_2  = [tf.gather(data, train_idxs, axis=0) for data in train_data]
y_train = np.take(PI_train, train_idxs, axis=0)
X_val_1, X_val_2 = [tf.gather(data, val_idxs, axis=0) for data in train_data]
y_val = np.take(PI_train, val_idxs, axis = 0)


test_data = [tf_data_test_1, tf_data_test_2]

name = "RipsNet"
history = train_model(model, [X_train_1, X_train_2],
                          [X_val_1, X_val_2], test_data, y_train, y_val, PI_test,
                          callback, name, epochs = args.epochs, logger = run)
bar.update()
bar.close()

run.stop()