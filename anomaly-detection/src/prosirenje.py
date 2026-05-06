import numpy as np

#Nystromovo prosirenje na sve tocke koje nisu u poduzorkovanju za stvaranje jezgre
#u radu (31) u nasem temeljnom radu, Fowlkes, Belongie, Chung, Malik, 2004

def prosiri_nystrom(X_podskup, X_ostatak, embedding, sv_vrijednosti, sigma, batch=500):
    n = X_ostatak.shape[0]
    d = embedding.shape[1]
    rezultat = np.zeros((n, d))


    #u podskupu je (m, f) piksela, u ostatku je (n, f) piksela
    #embedding je dakle oblika (m, d) - > moram to proširiti na ovih ostalih (n, d) embedding ostalih - to je shape od rezultat
    
    
    #da ne racunam odjednom za sve ostale piksele, radim po batchevima
    for start in range(0, n, batch): # svaki X_batch je oblika (<=500,f), znaci ja pokrivam ovih ostalih n po 500 po iteraciji u ovom... radim ovako da i slabija racunala s manje RAMa mogu izvesti, ako vam i dalje ne zeli izvesti ovu f-ju, smanjite batch size
        kraj = min(start + batch, n)
        X_batch = X_ostatak[start:kraj] #oblika (batch, f)

        #umjesto da radim razlika = X_batch[:, np.newaxis, :] - X_podskup[np.newaxis, :, :]
        #sto napravi tenzor oblika (batch, m, f), sto je 500 * 20 000 * 64 = 640 milijuna floatova (lol)
        #mogu napraviti ||a-b||^2 = ||a||^2 + ||b||^2 - 2a*b^T
        #sto je puno brze :)
        
        norma_batch = np.sum(X_batch**2, axis=1, keepdims=True)      #oblika (batch, 1)
        norma_podskup = np.sum(X_podskup**2, axis=1, keepdims=True)  #oblika (m, 1)
        udaljenosti_kvadrat = norma_batch + norma_podskup.T - 2 * X_batch @ X_podskup.T  #oblika (batch, m)
        udaljenosti_kvadrat = np.maximum(udaljenosti_kvadrat, 0)  #da bude numericki sigurno
        
        W_batch = np.exp(-udaljenosti_kvadrat / sigma[np.newaxis, :]**2)
        W_batch = W_batch / W_batch.sum(axis=1, keepdims=True)
        rezultat[start:kraj] = W_batch @ embedding / sv_vrijednosti[np.newaxis, :]
    
    return rezultat
    
def prosiri_laplacian_piramida(X_podskup, X_ostatak, embedding, max_razina=6, prag=1e-6, batch=50):
    n = X_ostatak.shape[0]
    m = X_podskup.shape[0]
    d = embedding.shape[1]
    rezultat = np.zeros((n, d))
    
    # epsilon_0 -pocetno velik br (jednadžba 10)
    uzorak = min(500, m)
    idx = np.random.choice(m, uzorak, replace=False)
    X_uzorak = X_podskup[idx]
    norma_u = np.sum(X_uzorak**2, axis=1, keepdims=True)
    dist_sq_uzorak = np.maximum(norma_u + norma_u.T - 2 * X_uzorak @ X_uzorak.T, 0)
    epsilon_0 = np.median(dist_sq_uzorak[dist_sq_uzorak > 0])
    
    norma_podskup = np.sum(X_podskup**2, axis=1, keepdims=True)  # (m, 1)
    ostatak_f = embedding.copy()
    
    for razina in range(max_razina):
        epsilon = epsilon_0 / (2**razina)  #jkednandžba 12
        
        # smoothing operator na podskupu (jednandžbe 11, 13, 14, 15)
        aproks_podskup = np.zeros((m, d))
        for start in range(0, m, batch):
            kraj = min(start + batch, m)
            X_batch = X_podskup[start:kraj]
            norma_batch = np.sum(X_batch**2, axis=1, keepdims=True)
            dist_sq = np.maximum(norma_batch + norma_podskup.T - 2 * X_batch @ X_podskup.T, 0)
            K = np.exp(-dist_sq / epsilon)           # jed 10/12
            K = K / (K.sum(axis=1, keepdims=True) + 1e-10)  #jed 11/13
            aproks_podskup[start:kraj] = K @ ostatak_f      #jed 14/15
        
        # prosiri na ostatak (17/18)
        for start in range(0, n, batch):
            kraj = min(start + batch, n)
            X_batch = X_ostatak[start:kraj]
            norma_batch = np.sum(X_batch**2, axis=1, keepdims=True)
            dist_sq = np.maximum(norma_batch + norma_podskup.T - 2 * X_batch @ X_podskup.T, 0)
            K = np.exp(-dist_sq / epsilon)           # eq 12
            K = K / (K.sum(axis=1, keepdims=True) + 1e-10)  #jed 13
            rezultat[start:kraj] += K @ ostatak_f           #jed 17/18
        
        # novi ostatak (jed 16)
        ostatak_f = ostatak_f - aproks_podskup
        
        if np.max(np.abs(ostatak_f)) < prag:
            print(f"  Laplacian piramida konvergirala na razini {razina}")
            break
    
    return rezultat