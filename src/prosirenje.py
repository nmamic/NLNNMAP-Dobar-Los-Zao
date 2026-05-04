import numpy as np

#Nystromovo prosirenje na sve tocke koje nisu u poduzorkovanju za stvaranje jezgre
#u radu (31) u nasem temeljnom radu, Fowlkes, Belongie, Chung, Malik, 2004

def prosiri_nystrom(X_podskup, X_ostatak, embedding, sv_vrijednosti, sigma):
    # na X_podskup imamo definirano, prosirujemo na X_ostatak

    #za svaku novu tocku y, Psi(y) = 1/sv_vr * \sum_i (W(y,x_i) * Psi(x_i))
    
    razlika = X_ostatak[:, np.newaxis, :] - X_podskup[np.newaxis, :, :] #(n-m, m)
    udaljenosti = np.sqrt(np.sum(razlika**2, axis = -1))
    
    #nemam sigme za tocke koje nisu u uzorkovanom podskupu, pa samo koristim te
    #w(y,x_i) = exp(-d(y,x_i)^2 / sigma[i]^2) (jednadzba 8)
    W_ostatak = np.exp(-udaljenosti**2 / sigma[np.newaxis, :] ** 2)

    sume_redovi = W_ostatak.sum(axis=1, keepdims=True)
    W_ostatak = W_ostatak / sume_redovi

    #Sad konacno nystromovo prosirenje:
    #prosirenje (n-m, m) @ (m, d) - > (n-m, d)
    embedding_ostatak = W_ostatak @ embedding / sv_vrijednosti[np.newaxis, :]

    return embedding_ostatak
