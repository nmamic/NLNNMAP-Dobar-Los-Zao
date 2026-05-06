import numpy as np

#trebam izracunati globalni sigma kako je opisano ispod jednadzbe (19) u radu
#uzimam n_pair parova piksela i racunam difuzijsku udaljenost za svaki par

def izrac_globalni_sigma(embedding, n_pair=1000, r=1):
    m = embedding.shape[0]
    indeks1 = np.random.randint(0, m, n_pair)
    indeks2 = np.random.randint(0, m, n_pair)
    difuzijske_udaljenosti = np.sum((embedding[indeks1] - embedding[indeks2])**2, axis=1)
    sigma_kvadrat = np.var(difuzijske_udaljenosti)
    print(f"  globalni_sigma: {r * sigma_kvadrat:.6f}")
    #return r * sigma_kvadrat #ovako je zadana u radu
    return max(r * sigma_kvadrat, 1e-3)

#racuna anomaly score (jednadzba 20)
#W je velicina prozora koji gledamo oko piksela (zapravo manhattan radijus)
#M je velicina maske oko piksela unutar prozora (opet manhattan radijus)
def izrac_anomaly_score(embedding, slika_shape, W, M):
    H, N = slika_shape
    globalni_sigma = izrac_globalni_sigma(embedding, r=20)
    scores = np.zeros(H * N)
    
    for i in range(H * N):
        red_i = i // N
        stup_i = i % N
        
        # samo pikseli u prozoru oko i
        red_min = max(0, red_i - W)
        red_max = min(H, red_i + W + 1)
        stup_min = max(0, stup_i - W)
        stup_max = min(N, stup_i + W + 1)
        
        redovi = np.arange(red_min, red_max)
        stupci = np.arange(stup_min, stup_max)
        rr, cc = np.meshgrid(redovi, stupci, indexing='ij')
        manhattan = np.abs(rr - red_i) + np.abs(cc - stup_i)
        
        maska = (manhattan > M) & (manhattan <= W)
        j_indeksi = (rr[maska] * N + cc[maska]).flatten()
        
        if len(j_indeksi) == 0:
            scores[i] = 1.0
            continue
        
        razlike = embedding[i] - embedding[j_indeksi]
        w = np.exp(-np.sum(razlike**2, axis=1) / globalni_sigma)
        scores[i] = 1 - w.mean()
    
    return scores.reshape(H, N)
