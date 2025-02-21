from itertools import combinations, combinations_with_replacement, product
import numpy as np
import torch

def zero_out_diagonal(distances): # make 0 on diagonal
    return distances * (np.ones_like(distances) - np.eye(*distances.shape))

def pdist_gpu(a, b, device = 'cuda:0'):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

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

def triplet_accuracy(input_data, latent_data, triplets=None):
    # calculate distance matricies
    input_data = input_data.reshape(input_data.shape[0], -1)
    input_distances = zero_out_diagonal(pdist_gpu(input_data, input_data))
    latent_data = latent_data.reshape(latent_data.shape[0], -1)
    latent_distances = zero_out_diagonal(pdist_gpu(latent_data, latent_data))
    # generate triplets
    if triplets is None:
        triplets = np.asarray(list(combinations(range(len(input_data)), r=3)))
    i_s = triplets[:, 0]
    j_s = triplets[:, 1]
    k_s = triplets[:, 2]
    acc = (np.logical_xor(
        input_distances[i_s, j_s] < input_distances[i_s, k_s], 
        latent_distances[i_s, j_s] < latent_distances[i_s, k_s]
    ) == False)
    acc = np.mean(acc.astype(np.int32))
    return acc
