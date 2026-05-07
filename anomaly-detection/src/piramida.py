from scipy.ndimage import gaussian_filter
import numpy as np

from jezgra import izrac_knn, izrac_lokalni_sigma, izrac_affinity, normaliz_po_retcima
from mape import konstr_difuz_preslikavanje
from prosirenje import prosiri_nystrom, prosiri_laplacian_piramida
from ocjene import izrac_anomaly_score

def izvuci_patcheve(slika, indeksi, patch_velicina):
    H,W = slika.shape
    pola = patch_velicina //2

    #paddam sliku za granicne piksele
    padded = np.pad(slika, pola, mode='reflect')

    patchevi = []
    for indeks in indeksi:
        red = indeks // W
        stupac = indeks % W

        patch = padded[red:red + patch_velicina, stupac:stupac + patch_velicina]
        assert patch.shape == (patch_velicina, patch_velicina), \
            f"Patch shape {patch.shape} != ({patch_velicina}, {patch_velicina}) at red={red}, stupac={stupac}"
        patchevi.append(patch.flatten())

    return np.array(patchevi) #oblika (len(indeksi), patch_velicina)

def gaussovska_piramida(slika, br_razina):
    piramida = [slika.copy()]  # start with original
    trenutno = slika.copy()
    
    for _ in range(br_razina):
        zamuceno = gaussian_filter(trenutno, sigma=1) #primjenim filter da nemam aliasing pri poduzorkovanju
        poduzorkovano = zamuceno[::2, ::2]  # podutorkujem svaki drugi piksel u obje dimenzije
        piramida.append(poduzorkovano)
        trenutno = poduzorkovano

    #i na kraju obrnem redoslijed da je najgrublja prva
    return piramida[::-1]

def uzorkovanje(slika, sus_prosli, m_ukupno, m_sus_postotak=0.1):
    #m_sus = postotak sumnjivih piksela u uzorkovanju na trenutnoj razini (npr kao u tablici 1 iz rada)
    #m_ukupno = ukupno piksela na ovoj razini
    #sus_prosli = sumnjivi pikseli iz prosle razine
    H, W = slika.shape
    svi_indeksi = np.arange(H * W)

    if sus_prosli is None:
        #ovo je na prvoj razini, kad nemamo proslu, onda su svi pikseli random
        return np.random.choice(svi_indeksi, m_ukupno, replace=False)
    
    #sus_prosli je boolean maska
    #i to za proslu sliku, dakle manje rezolucije - ja je tu upscaleam
    #na nacin da 1 sumnjivi piksel u manjoj rezoluciji pretvorim u 2x2 blok sumnjivih piskela u vecoj rezoluciji
    #ovo je nuzno da bi povecao vjerojatnost da uzorkujem sumnjiva podrucja, jer su neke anomalije jako male, npr 15x3 u punoj rezoluciji
    #to radim s np repeat, koje npr radi (repeat([1,2],2) = [1, 1, 2, 2])
    sus_upscale = np.repeat(np.repeat(sus_prosli, 2, axis=0), 2, axis=1)
    #where uvijek vrati tuple, gdje je prvi element tuple-a array indeksa elemenata koji su True
    sus_indeksi = np.where(sus_upscale.flatten())[0]

    #sad biram sumnjive
    m_sus = min(len(sus_indeksi), int(m_ukupno * m_sus_postotak))
    odabrani_sus = np.random.choice(sus_indeksi, m_sus, replace=False)

    m_preostali = m_ukupno - m_sus
    nesumnjivi = np.setdiff1d(svi_indeksi, sus_indeksi)
    odabrani_ostali = np.random.choice(nesumnjivi, m_preostali, replace=False)
    
    return np.concatenate([odabrani_sus, odabrani_ostali])


def run_pyramid(slika, config):
    piramida = gaussovska_piramida(slika, config.n_razina)
    sumnjivi = None
    
    for razina, img in enumerate(piramida):
        H,W = img.shape
        n_piksela = H*W
        m_ukupno = int(config.uzorkovanje_postotci[razina] * n_piksela)
        print(f"Razina {razina}: img={img.shape}, sumnjivi shape={sumnjivi.shape if sumnjivi is not None else None}")
        podskup_indeksi = uzorkovanje(img, sumnjivi, m_ukupno)
        print(f"Razina {razina}: slika {H}x{W}, m_ukupno={m_ukupno}")
        
        # samo patchevi za podskup
        X_podskup = izvuci_patcheve(img, podskup_indeksi, config.velicine_patcha[razina])
        novi_indeksi = np.where(~np.isin(np.arange(n_piksela), podskup_indeksi))[0]
        X_novi = izvuci_patcheve(img, novi_indeksi, config.velicine_patcha[razina])
        
        #sad na podskupu napravim korake za difuzijsko preslikavanje
        udaljenosti, indeksi = izrac_knn(X_podskup, config.k)
        sigma = izrac_lokalni_sigma(udaljenosti)
        W_aff = izrac_affinity(X_podskup, config.k)
        P = normaliz_po_retcima(W_aff)
        embedding, sv_vrijednosti = konstr_difuz_preslikavanje(P, config.d_po_razini[razina], config.t)
        
        #i sad prosirim na sve piksele
        if config.koristiti_laplacian:
            embedding_novo = prosiri_laplacian_piramida(X_podskup, X_novi, embedding)
        else:
            embedding_novo = prosiri_nystrom(X_podskup, X_novi, embedding, sv_vrijednosti, sigma)
        
        #sad potpuni embedding:
        embedding_cijeli = np.zeros((n_piksela, config.d_po_razini[razina]))
        embedding_cijeli[podskup_indeksi] = embedding
        embedding_cijeli[novi_indeksi] = embedding_novo
        
        #izracunam anomaly scoreove
        scores = izrac_anomaly_score(embedding_cijeli, img.shape, config.W_po_razini[razina], config.M_po_razini[razina])

        print(f"  Raspon scoreova: {scores.min():.4f} - {scores.max():.4f}")
        print(f"  99. centil score-ova: {np.percentile(scores, 99):.4f}")
        print(f"  50. centil score-ova: {np.percentile(scores, 50):.4f}")
        
        #kao u radu uzimam 95ti centil
        sumnjivi = scores > np.percentile(scores, 95)
    
    return scores
