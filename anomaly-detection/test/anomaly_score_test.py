import os
import sys
from pathlib import Path
import numpy as np
from scipy.linalg import eigh
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import rbf_kernel
from scipy.optimize import brentq
from scipy.stats import gamma
from PIL import Image
import csv
import time
import traceback
from datetime import datetime

SRC_PATH = Path(os.path.dirname(Path.cwd())) / "anomaly-detection/src"
sys.path.insert(0, str(SRC_PATH))

from config import Config
from piramida import run_pyramid


def ucitaj_sliku(putanja, ciljana_velicina=(200, 200)):
    slika = Image.open(putanja)
    
    #pretvori u grayscale ako nije
    if slika.mode != 'L':
        slika = slika.convert('L')
    
    #smanji na 200x200 kao u radu
    slika = slika.resize(ciljana_velicina, Image.LANCZOS)
    
    #pretvori u numpy i normaliziraj
    slika_array = np.array(slika, dtype=np.float64) / 255.0
    
    return slika_array

def ng_beva(X, max_iter=50, gamma_val=1.25, tol=0.01):
    """
    NG BEVA - Background Parameter Estimation

    Parameters:
        X : ndarray (N x p)  -> ulazni podaci (pikseli)
        max_iter : int       -> maksimalan broj iteracija
        gamma_val : float    -> parametar γ
        tol : float          -> tolerancija za zaustavljanje

    Returns:
        dict sa procijenjenim parametrima
    """

    N, p = X.shape

    # Čuvamo indekse originalnih podataka
    original_idx = np.arange(N)

    # Inicijalizacija
    A_c = np.array([], dtype=int)      # anomalni indeksi
    B_c = X.copy()                     # background podaci
    B_idx = original_idx.copy()        # indeksi background podataka
    tau = np.nan

    w = np.ones(len(B_c))
    d0 = (np.sqrt(p) + np.sqrt(2))**2

    i = 0
    izbaceno_posto=0.01

    while i < max_iter and len(B_c) > 0:

        # 1. Robustna procjena sredine i kovarijacijske matrice
        w_sum = np.sum(w)

        mu = np.sum(w[:, None] * B_c, axis=0) / w_sum

        diff = B_c - mu

        w2 = w ** 2
        Sigma = (diff.T @ (w2[:, None] * diff)) / (np.sum(w2) - 1)

        # Regularizacija radi numeričke stabilnosti
        Sigma += 1e-6 * np.eye(p)

        # 2. Mahalanobis distance
        d = np.array([
            (x - mu).T @ np.linalg.solve(Sigma, (x - mu))
            for x in B_c
        ])

        # 3. Update težina
        w_new = np.where(
            d <= d0,
            1,
            (d0 / (d + 1e-8)) *
            np.exp(-0.5 * ((d - d0) ** 2) / (gamma_val ** 2))
        )

        # 4. Gamma fitting (MLE)
        k_hat, loc, theta_hat = gamma.fit(d + 1e-8, floc=0)

        # 5. Threshold 

        tau =  gamma.ppf(0.99, k_hat, scale=theta_hat)

        

        # Lokalni indeksi anomalija u trenutnom B_c
        idx = np.where(d >= tau)[0]

        if len(idx) == 0:
            break

        # Spremi ORIGINALNE indekse anomalija
        A_c = np.concatenate([A_c, B_idx[idx]])

        # Ukloni anomalije iz background skupa
        mask = np.ones(len(B_c), dtype=bool)
        mask[idx] = False

        B_c = B_c[mask]
        B_idx = B_idx[mask]
        w = w_new[mask]

        i += 1

        # Stopping kriterij
        izbaceno_posto = len(idx) / len(B_c) 

        if izbaceno_posto < tol:
            break

    return {
        "A_c": A_c,          # indeksi anomalija u originalnom X
        "B_c": B_idx,        # indeksi background elemenata u originalnom X
        "mu": mu,
        "Sigma": Sigma,
        "k_hat": k_hat,
        "theta_hat": theta_hat,
        "tau": tau
    }

def clustering_nystrom(X, L=3, sigma=1.0, m=500):
    """
    Spectral clustering using Nyström approximation.

    Parameters
    ----------
    X : ndarray (N x p)
        Data matrix

    L : int
        Number of clusters

    sigma : float
        RBF kernel scale

    m : int
        Number of landmark points

    Returns
    -------
    labels : ndarray (N,)
        Cluster assignments
    """

    N = X.shape[0]

    # ---------------------------------------------------------
    # 1. Random landmark selection
    # ---------------------------------------------------------
    landmark_idx = np.random.choice(N, m, replace=False)

    X_landmarks = X[landmark_idx]

    # ---------------------------------------------------------
    # 2. Compute A matrix (m x m)
    # ---------------------------------------------------------
    #
    # Similarity among landmarks
    #
    gamma = 1.0 / (2.0 * sigma**2)

    A = rbf_kernel(
        X_landmarks,
        X_landmarks,
        gamma=gamma
    )

    # Stabilization
    A += 1e-8 * np.eye(m)

    # ---------------------------------------------------------
    # 3. Compute B matrix (m x N)
    # ---------------------------------------------------------
    #
    # Similarity between landmarks and ALL points
    #
    B = rbf_kernel(
        X_landmarks,
        X,
        gamma=gamma
    )

    # ---------------------------------------------------------
    # 4. Eigendecomposition of A only
    # ---------------------------------------------------------
    eigvals, eigvecs = eigh(A)

    # Take largest L eigenvalues
    eigvals = eigvals[-L:]
    eigvecs = eigvecs[:, -L:]

    # Numerical stability
    eigvals = np.maximum(eigvals, 1e-8)

    # ---------------------------------------------------------
    # 5. Nyström extension
    # ---------------------------------------------------------
    #
    # Approximate eigenvectors for ALL points
    #
    V = B.T @ eigvecs @ np.diag(1.0 / np.sqrt(eigvals))

    # ---------------------------------------------------------
    # 6. Row normalization
    # ---------------------------------------------------------
    norms = np.linalg.norm(V, axis=1, keepdims=True) + 1e-8
    V_norm = V / norms

    # ---------------------------------------------------------
    # 7. KMeans in spectral space
    # ---------------------------------------------------------
    kmeans = KMeans(
        n_clusters=L,
        n_init=10,
        random_state=0
    )

    labels = kmeans.fit_predict(V_norm)

    return labels

def detect(slika, br_klastera=3, sigma=10.0, m_za_poduzorak=500, max_iter=50, gamma_val=1.25, tol=0.01, ispis=True):
    # slika.size kod PIL slika vraća (širina, visina)
    h, w, _ = slika.shape
    podaci = np.array(slika).reshape(-1, 3)
    
    rezultati = {}
    heatmap_mahalanobis = np.zeros(len(podaci))
    detekcija_maska = np.zeros(len(podaci))
    anomaly_score = np.zeros(len(podaci))
    
    klasteri = clustering_nystrom(podaci, br_klastera, sigma, m_za_poduzorak)
    klasteri = np.array(klasteri)

    for i in np.unique(klasteri):
        idx_maska=np.where(klasteri == i)[0].astype(int) 
        X_fit = podaci[idx_maska]
    
        rezultat = ng_beva(X_fit, max_iter=max_iter, gamma_val=gamma_val, tol=tol)
        
        mu = rezultat["mu"]
        Sigma = rezultat["Sigma"]
        
        # Centriranje podataka (X - mu)
        X_centered = X_fit - mu
        
        # Rješavanje sustava Sigma * z = X_centered.T
        # Mahalanobis^2 = (X-mu).T * inv(Sigma) * (X-mu) 
        # Možemo izračunati z = inv(Sigma) * (X-mu).T rješavanjem sustava:
        try:
            # solve rješava AX = B, ovdje je A=Sigma, B=X_centered.T
            z = np.linalg.solve(Sigma, X_centered.T) 
            # Udaljenost je suma umnoška centriranih podataka i rješenja sustava (z)
            udaljenosti = np.sqrt(np.sum(X_centered * z.T, axis=1))
            heatmap_mahalanobis[idx_maska] = udaljenosti
            anomaly_score[idx_maska] = udaljenosti / rezultat["tau"]
        except np.linalg.LinAlgError:
            # U slučaju singularne matrice (npr. premalo točaka), koristi fallback
            heatmap_mahalanobis[idx_maska] = 0
            anomaly_score[idx_maska] = 0

        # Mapiranje detekcije
        A_c_izvorni = idx_maska[rezultat["A_c"].astype(int) ]
        B_c_izvorni = idx_maska[rezultat["B_c"].astype(int) ]
        
        detekcija_maska[A_c_izvorni] = 1 
        detekcija_maska[B_c_izvorni] = 0

        rezultati[i] = {
            "A_c": A_c_izvorni,
            "B_c": B_c_izvorni,
            "mu": mu,
            "Sigma": Sigma,
            "k_hat": rezultat["k_hat"],
            "theta_hat": rezultat["theta_hat"]
        }

    if ispis:
        # Iscrtavanje (reshape na h, w jer imshow očekuje Visina x Širina)
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))
        ax[0, 0].imshow(np.array(slika))
        ax[0, 0].set_title("Original")
        
        ax[0, 1].imshow(klasteri.reshape(h, w), cmap='tab10')
        ax[0, 1].set_title("Klasteri")
        
        im = ax[1, 0].imshow(heatmap_mahalanobis.reshape(h, w), cmap='hot')
        ax[1, 0].set_title("Mahalanobis Heatmap")
        plt.colorbar(im, ax=ax[1, 0])
        
        viz_maska = np.zeros((len(detekcija_maska), 3))
        viz_maska[detekcija_maska == 1] = [1, 0, 0]  # Crvena za A_c (Anomalije)
        viz_maska[detekcija_maska == 0] = [0, 0, 1]  # Plava za B_c (Pozadina)
        
        ax[1, 1].imshow(viz_maska.reshape(h, w))
        ax[1, 1].set_title("Crveno: Anomalije (A_c), Plavo: Pozadina (B_c)")
        
        for a in ax.flat: a.axis('off')
        plt.tight_layout()
        plt.show()

    return {
        "rezultati": rezultati,
        "maska": detekcija_maska.reshape(h, w),
        "heatmap": heatmap_mahalanobis.reshape(h, w),
        "score": anomaly_score.reshape(h, w)
    }

def pixel_value_mean_std(img: np.ndarray, area_size: int = 3) -> np.ndarray:
    """
    Takes a black-and-white / grayscale image as a 2D NumPy array and returns
    a 3D array where each pixel contains:

        [pixel_value, local_area_mean, local_area_std]

    Parameters
    ----------
    img : np.ndarray
        2D grayscale image array of shape (height, width).
    area_size : int
        Size of the square local area around each pixel.
        Must be odd, e.g. 3, 5, 7.

    Returns
    -------
    np.ndarray
        Array of shape (height, width, 3).
    """

    if img.ndim != 2:
        raise ValueError("Input image must be a 2D grayscale array.")

    if area_size % 2 == 0:
        raise ValueError("area_size must be odd, e.g. 3, 5, 7.")

    img = img.astype(float)

    pad = area_size // 2

    padded = np.pad(img, pad_width=pad, mode="reflect")

    windows = sliding_window_view(padded, (area_size, area_size))

    local_mean = windows.mean(axis=(-1, -2))
    local_std = windows.std(axis=(-1, -2))

    result = np.stack([img, local_mean, local_std], axis=-1)

    return result
#================================================

#Ships dataset
images_dir = Path.cwd() / "data/ships"
images_paths = [i for i in os.listdir("data/ships")
                if not i.endswith("true.jpg")]
images = {i: ucitaj_sliku(images_dir / i, (200, 200)) 
          for i in images_paths}


#Laplacian - Anomaly Score
config = Config(koristiti_laplacian=True, r=10)
for i, slika in images.items():
    anomaly_score= run_pyramid(slika, config)
    np.save("test/output/ships/lap_anom_" + i.removesuffix(".jpg"),
            anomaly_score)

#Laplacian - Saliency Score
config = Config(koristiti_laplacian=True,
                koristiti_saliency_score=True,
                r=10)
for i, slika in images.items():
    anomaly_score= run_pyramid(slika, config)
    np.save("test/output/ships/lap_sal_" + slika.removesuffix(".jpg"),
            anomaly_score)

for i, slika in images.items():
    ng_rez = detect(pixel_value_mean_std(slika, 5), ispis=False)
    np.save("test/output/ships/ng_score_" + i.removsuffix(".jpg"),
            ng_rez["score"])
    np.save("test/output/ships/ng_mask_" + i.removesuffix(".jpg"),
            ng_rez["maska"])

#=============================================================================

#Shipwrecks dataset
images_dir = Path.cwd() / "data/shipwrecks"
images_paths = [i for i in os.listdir(images_dir)
          if i.endswith("500.jpg") and not i.startswith("Art")]
images = {i: ucitaj_sliku(images_dir / i, (200, 200)) 
          for i in images_paths}

#Laplacian - Anomaly Score
config = Config(koristiti_laplacian=True, uzorkovanje_postotci = [0.50, 0.35, 0.20])
for i, slika in images.items():
    anomaly_score= run_pyramid(slika, config)
    np.save("test/output/shipwrecks/lap_anom_" + i.removesuffix(".jpg"),
            anomaly_score)

#Nystrom - Anomaly Score
config = Config(koristiti_laplacian=False, uzorkovanje_postotci = [0.50, 0.35, 0.20])
for i, slika in images.items():
    anomaly_score = run_pyramid(slika, config)
    np.save("test/output/shipwrecks/nys_anom_" + i.removesuffix(".jpg"),
               anomaly_score)

#Laplacian - Saliency Score
config = Config(koristiti_laplacian=True,
                koristiti_saliency_score=True)
for i, slika in images.items():
    anomaly_score= run_pyramid(slika, config)
    np.save("test/output/shipwrecks/lap_sal_" + i.removesuffix(".jpg"),
            anomaly_score)

#Nystrom - Saliency Score
config = Config(koristiti_laplacian=False,
                koristiti_saliency_score=True)
for i, slika in images.items():
    anomaly_score = run_pyramid(slika, config)
    np.save("test/output/shipwrecks/nys_sal_" + i.removesuffix(".jpg"),
               anomaly_score)

for i, slika in images.items():
    ng_rez = detect(pixel_value_mean_std(slika, 5), ispis=False)
    np.save("test/output/shipwrecks/ng_score_" + i.removsuffix(".jpg"),
            ng_rez["score"])
    np.save("test/output/shipwrecks/ng_mask_" + i.removesuffix(".jpg"),
            ng_rez["maska"])

slika = ucitaj_sliku(images_dir / "Artificial_Reef_03_500.jpg", (200, 200))
anomaly_score = run_pyramid(slika, Config())
saliency_score = run_pyramid(slika, Config(koristiti_saliency_score=True))
np.save("test/output/shipwrecks/lap_anom_" + "Artificial_Reef_03_500",
        anomaly_score)
np.save("test/output/shipwrecks/lap_sal_" + "Artificial_Reef_03_500",
        saliency_score)

#==============================================================================

#Mines dataset
images_dir = Path.cwd() / "data/mines"
images_paths = [i for i in os.listdir(images_dir)
          if i.endswith("2021.jpg")]
images = {i: ucitaj_sliku(images_dir / i, (168, 168)) 
          for i in images_paths}

#Laplacian - Anomaly Score
config = Config(koristiti_laplacian=True)
for i, slika in images.items():
    anomaly_score= run_pyramid(slika, config)
    np.save("test/output/mines/lap_anom_" + i.removesuffix(".jpg"),
            anomaly_score)
    
#Nystrom - Anomaly Score
config = Config(koristiti_laplacian=False)
for i, slika in images.items():
    anomaly_score = run_pyramid(slika, config)
    np.save("test/output/mines/nys_anom_" + i.removesuffix(".jpg"),
               anomaly_score)

#Laplacian - Saliency Score
config = Config(koristiti_laplacian=True,
                koristiti_saliency_score=True)
for i, slika in images.items():
    anomaly_score= run_pyramid(slika, config)
    np.save("test/output/mines/lap_sal_" + i.removesuffix(".jpg"),
            anomaly_score)

#Nystrom - Saliency Score
config = Config(koristiti_laplacian=False,
                koristiti_saliency_score=True)
for i, slika in images.items():
    anomaly_score = run_pyramid(slika, config)
    np.save("test/output/mines/nys_sal_" + i.removesuffix(".jpg"),
               anomaly_score)

for i, slika in images.items():
    ng_rez = detect(pixel_value_mean_std(slika, 5), ispis=False)
    np.save("test/output/mines/ng_score_" + i.removsuffix(".jpg"),
            ng_rez["score"])
    np.save("test/output/mines/ng_mask_" + i.removesuffix(".jpg"),
            ng_rez["maska"])