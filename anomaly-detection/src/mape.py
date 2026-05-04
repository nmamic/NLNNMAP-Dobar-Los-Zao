from scipy.sparse.linalg import eigs
import numpy as np

def konstr_difuz_preslikavanje(P, d=6, t=1):
    #d = dimenzija u koju ulažemo
    #t = br koraka u markovljevu lancu

    sv_vrijednosti, sv_vektori = eigs(P, k=d+1, which='LM')

    sv_vrijednosti = sv_vrijednosti.real
    sv_vektori = sv_vektori.real

    idx = np.argsort(-sv_vrijednosti)
    sv_vrijednosti = sv_vrijednosti[idx]
    sv_vektori = sv_vektori[:, idx]

    sv_vrijednosti = sv_vrijednosti[1:] #odbacujem prvi sv vektor, pridružen sv vrijednosti 1
    sv_vektori = sv_vektori[:, 1:]

    ulaganje = sv_vektori * (sv_vrijednosti**t)

    return ulaganje, sv_vrijednosti
