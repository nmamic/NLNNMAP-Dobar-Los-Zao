import numpy as np
from scipy.stats import gamma
from sklearn.cluster import KMeans
from scipy.linalg import eigh
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from PIL import Image

def ng_beva(X, max_iter=50, gamma_val=1.25, tol=1e-6):
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

    # Inicijalizacija
    A_c = np.zeros((0, p))        # anomalije
    B_c = X.copy()                # background
    w = np.ones(len(B_c))         # težine
    d0 = np.sqrt(p) + np.sqrt(2)  # prag
    i = 0

    while i < max_iter and len(B_c) > 0:

        # 1. Robustna procjena sredine i kovarijanse
        w_sum = np.sum(w)
        mu = np.sum(w[:, None] * B_c, axis=0) / w_sum

        diff = B_c - mu
        Sigma = (diff.T @ (w[:, None] * diff)) / (np.sum(w**2) - 1)

        # Regularizacija (numerička stabilnost)
        Sigma += 1e-6 * np.eye(p)
        Sigma_inv = np.linalg.inv(Sigma)

        # 2. Mahalanobis distance
        d = np.array([
            (x - mu).T @ Sigma_inv @ (x - mu)
            for x in B_c
        ])

        # 3. Update težina
        w_new = np.where(
            d <= d0,
            1,
            (d0 / d) * np.exp(-0.5 * ((d - d0)**2) / (gamma_val**2))
        )

        # 4. Gamma fitting (MLE)
        k_hat, loc, theta_hat = gamma.fit(d, floc=0)

        # 5. Update skupova
        # prag τᶦ (možeš ga prilagoditi, npr. percentil)
        tau = gamma.ppf(0.99, k_hat, scale=theta_hat)

        idx = np.where(d >= tau)[0]

        if len(idx) == 0:
            break

        A_c = np.vstack([A_c, B_c[idx]])
        B_c = np.delete(B_c, idx, axis=0)
        w = np.delete(w_new, idx)

        i += 1

        # Stopping kriterij
        if len(idx) < tol:
            break

    return {
        "A_c": A_c,
        "B_c": B_c,
        "mu": mu,
        "Sigma": Sigma,
        "k_hat": k_hat,
        "theta_hat": theta_hat
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


def detect(slika, br_klastera=3, sigma=10.0, m_za_poduzorak=500, max_iter=50, gamma_val=1.25, tol=0.1, ispis=True):
    # slika.size kod PIL slika vraća (širina, visina)
    w, h = slika.size
    podaci = np.array(slika).reshape(-1, 3)
    
    rezultati = {}
    heatmap_mahalanobis = np.zeros(len(podaci))
    detekcija_maska = np.zeros(len(podaci))
    
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
        except np.linalg.LinAlgError:
            # U slučaju singularne matrice (npr. premalo točaka), koristi fallback
            heatmap_mahalanobis[idx_maska] = 0

        # Mapiranje detekcije
        A_c_izvorni = idx_maska[rezultat["A_c"].astype(int) ]
        B_c_izvorni = idx_maska[rezultat["B_c"].astype(int) ]
        
        detekcija_maska[A_c_izvorni] = 1 
        detekcija_maska[B_c_izvorni] = 2

        rezultati[i] = {
            "A_c": A_c_izvorni,
            "B_c": B_c_izvorni,
            "mu": mu,
            "Sigma": Sigma,
            "k_hat": rezultat["k_hat"],
            "theta_hat": rezultat["theta_hat"]
        }

    return {
        "rezultati": rezultati,
        "heatmap_mahalanobis": heatmap_mahalanobis.reshape(h, w),
        "detekcija_maska": detekcija_maska.reshape(h, w),
        "klasteri": klasteri.reshape(h, w)
    }
