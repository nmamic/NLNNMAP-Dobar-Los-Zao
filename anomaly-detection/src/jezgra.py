import numpy as np 
from scipy.sparse import lil_matrix
from scipy.sparse import diags

from sklearn.neighbors import NearestNeighbors

def izrac_knn(X, k=16):
    nn = NearestNeighbors(n_neighbors=k+1, metric='euclidean', algorithm='ball_tree')
    nn.fit(X)
    udaljenosti, indeksi = nn.kneighbors(X)
    return udaljenosti[:, 1:], indeksi[:, 1:]

def izrac_lokalni_sigma(udaljenosti, K = 7):
    sigma = udaljenosti[:, K-1]
    return np.maximum(sigma, 1e-10) #udaljenosti K-tih (0-indeksirano) točaka od trenutne točke (prvi indeks)

def izrac_affinity(X, k = 7):
    udaljenosti, indeksi = izrac_knn(X,k)
    lokalni_sigma = izrac_lokalni_sigma(udaljenosti)

    m = X.shape[0]
    W = lil_matrix((m,m)) # efikasno za inkrementirajuću izgradnju rijetke matrice

    for i in range(m):
        for j_indeks in range(k):
            j = indeksi[i,j_indeks]
            d = udaljenosti[i,j_indeks]

            w = np.exp(-d**2 / (lokalni_sigma[i] * lokalni_sigma[j]))
            W[i, j] = w
            W[j, i] = w # osiguram simetričnost


    return W.tocsr() #compressed sparse row format puno brzi za aritmeticke i matricno vektorske operacije

def normaliz_po_retcima(W):
    sume_redovi = np.array(W.sum(axis=1)).flatten()
    jedan_kroz_d = diags(1.0 / sume_redovi)
    P = jedan_kroz_d @ W
    return P
        

