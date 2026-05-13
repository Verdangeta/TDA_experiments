import tensorflow as tf
import ot
import torch
import numpy as np
from scipy.stats import entropy
from tqdm import tqdm
from sklearn.metrics import pairwise_distances

gaus_kernel = kernel = lambda x: np.exp(-np.sum(x**2,axis = 1)/2)/(2*np.pi)

# Symetrical KL distence, which we use as loss function.
def sym_KL(y_true, y_pred):
    loss = tf.keras.losses.KLDivergence()
    return (loss(y_true, y_pred) + loss(y_pred, y_true))/2


def K_H(h, kernel, u):
    return  kernel(u/h)/(h**2)
    

## For estimatinng optimal bandwidth parameter we need to calculate K_H between all our diagrams.
def calculate_K_Hs(
                diagrams,
                kernel,
                h = 0.1):
    
    N = len(diagrams)
    result = np.empty((N, N), dtype=object)
    result_solo = np.empty((N, N), dtype=object)

    for i in range(N):
        for j in range(N):
            
            res = []
            res_solo = []
            for point in diagrams[i]:
                res.append(K_H(h, kernel, point-diagrams[j]))
                res_solo.append(K_H(h, kernel, point))
                
            res = np.stack(res)
            result[i,j] = res.copy()
            result_solo[i, j] = res_solo.copy()
    return result, result_solo
            

# Special DenseRagged layer for processing 2d-tensors with not fixed first dimmension
class DenseRagged(tf.keras.layers.Layer):
    def __init__(self, units, use_bias=True, activation='linear', **kwargs):
        super(DenseRagged, self).__init__(**kwargs)
        self._supports_ragged_inputs = True 
        self.units = units
        self.use_bias = use_bias
        self.activation = tf.keras.activations.get(activation)
    def build(self, input_shape):
        last_dim = input_shape[-1]
        self.kernel = self.add_weight('kernel', shape=[last_dim, self.units], trainable=True)
        if self.use_bias:
            self.bias = self.add_weight('bias', shape=[self.units,], trainable=True)
        else:
            self.bias = None
        super(DenseRagged, self).build(input_shape)
    def call(self, inputs):
        outputs = tf.ragged.map_flat_values(tf.matmul, inputs, self.kernel)
        if self.use_bias:
            outputs = tf.ragged.map_flat_values(tf.nn.bias_add, outputs, self.bias)
        outputs = tf.ragged.map_flat_values(self.activation, outputs)
        return outputs
        
# The next layer after DenseRagged to transoform not fixed tennsor innto features vector.
class PermopRagged(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(PermopRagged, self).__init__(**kwargs)
        self._supports_ragged_inputs = True 
    def build(self, input_shape):
        super(PermopRagged, self).build(input_shape)
    def call(self, inputs):
        out = tf.math.reduce_sum(inputs, axis=1)
        return out





# Calculating distance matrix on gpu
def pdist_gpu(a, b, device = 'cuda:0'):
    A = torch.tensor(a, dtype = torch.float64)
    B = torch.tensor(b, dtype = torch.float64)

    size = (A.shape[0] + B.shape[0]) * A.shape[1] / 1e9
    max_size = 0.2

    if size > max_size:
        parts = int(size / max_size) + 1
    else:
        parts = 1

    pdist = np.zeros((A.shape[0], B.shape[0]))
    At = A.to(device)

    for p in range(parts):
        i1 = int(p * B.shape[0] / parts)
        i2 = int((p + 1) * B.shape[0] / parts)
        i2 = min(i2, B.shape[0])

        Bt = B[i1:i2].to(device)
        pt = torch.cdist(At, Bt)
        pdist[:, i1:i2] = pt.cpu()

        del Bt, pt
        torch.cuda.empty_cache()

    del At

    return pdist

def sep_dist(a, b, pdist_device = 'cpu'):
    if pdist_device == 'cpu':
        d1 = pairwise_distances(b, a, n_jobs = 40)
        d2 = pairwise_distances(b, b, n_jobs = 40)
    else:
        d1 = pdist_gpu(b, a, device = pdist_device)
        d2 = pdist_gpu(b, b, device = pdist_device)

    s = a.shape[0] + b.shape[0]

    apr_d = np.zeros((s, s))
    apr_d[a.shape[0]:, :a.shape[0]] = d1
    apr_d[a.shape[0]:, a.shape[0]:] = d2

    return apr_d

# My functionn for calculated distence between two measures on square 1x1 with different methods.
def measure_dist(dist_initial, dist_predicted, resolution = 50, method = "Wasserstein", epsilon = 1e-8, verbose = False):
    assert resolution == np.sqrt(len(dist_initial[0])) and resolution == np.sqrt(len(dist_predicted[0])), "Wrong shape!!"

    distances = []
    
    if method == "Wasserstein":
        x = np.linspace(0, 1, resolution)
        y = np.linspace(0, 1, resolution)
        X,Y = np.meshgrid(x,y)
    
        coorinates = np.stack((X.flatten(), Y.flatten()), axis = 1)
        
        M = ot.dist(coorinates, coorinates)
        M /= M.max()
        
        for i in tqdm(range(len(dist_predicted)), disable = ~verbose):
            xt = np.float64(dist_predicted[i])
        
            xs = np.float64(dist_initial[i])
            
            xs = xs/xs.sum()
            xt = xt/xt.sum()

            xt = np.clip(xt, epsilon, None)
            xs = np.clip(xs, epsilon, None)
            
            n = len(xt)
            
            # Calculate 2D EMD (Wasserstein)
            w_distance = ot.emd2(xt, xs, M)
            distances.append(w_distance)
    elif method == "KL":
        for i in tqdm(range(len(dist_predicted)), disable = ~verbose):
            q = np.float64(dist_predicted[i]).flatten()
        
            p = np.float64(dist_initial[i]).flatten()

            p = p/p.sum()
            q = q/q.sum()

            p = np.clip(p, epsilon, None)
            q = np.clip(q, epsilon, None)

            kl_divergence = entropy(p, q)
            distances.append(kl_divergence)
    elif method == "KL_sym":
        for i in tqdm(range(len(dist_predicted)), disable = ~verbose):
            q = np.float64(dist_predicted[i]).flatten()
        
            p = np.float64(dist_initial[i]).flatten()

            p = p/p.sum()
            q = q/q.sum()

            p = np.clip(p, epsilon, None)
            q = np.clip(q, epsilon, None)

            kl_divergence = (entropy(p, q) + entropy(q, p))/2
            distances.append(kl_divergence)
    if verbose == True:
        print(f"Mean {method} distance: {np.mean(distances)}")
    return distances
    
### Realization of bandwidht estimation from paper
def estimate_optimal_bandwidth(diagrams,
                               kernel,
                               K_Hs, K_H_solo,
                               h_list = [0.1, 0.2, 0.3]):
    N = len(diagrams)
    
    right_part = 0
    left_part = 0
    
    for i in range(N):
        for j in range(N):
            if j!=i:
                n_i = diagrams[i].shape[0]
                n_j = diagrams[j].shape[0]
                p_i_j = np.sum(K_Hs[i , j])/(n_i*n_j)
            
        right_part += p_i_j

    right_part = -right_part * 2/(N*(N-1))
    
    

