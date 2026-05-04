import numpy as np 
from scipy.sparse import lil_matrix
from scipy.sparse import diags

def izrac_knn(X, k = 7):
    razlike = X[:, np.newaxis, :] - X[np.newaxis, :, :]; # (m x 1 x f) - (1 x m x f), reducira se na m x m x f, razlike[i,j] = x_i - x_j
    udaljenosti = np.sqrt(np.sum(razlike**2, axis=-1))

    indeksi = np.argsort(udaljenosti, axis=1)[:, 1:k+1]
    udaljenosti = np.take_along_axis(udaljenosti, indeksi, axis=1)

    return udaljenosti, indeksi

def izrac_lokalni_sigma(udaljenosti, K = 7):
    return udaljenosti[:, K-1] #udaljenosti K-tih (0-indeksirano) točaka od trenutne točke (prvi indeks)

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
        

