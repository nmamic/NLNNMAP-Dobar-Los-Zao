from types import FunctionType
import numpy as np
from numpy.typing import NDArray

def sigmoid(podaci: NDArray) -> NDArray:
    return 1 / (1 + np.exp(-podaci))

def sigmoid_derivacija(podaci: NDArray) -> NDArray:
    return sigmoid(podaci) * (1- sigmoid(podaci))

def reLU(podaci: NDArray) -> NDArray:
    return np.maximum(0,podaci)

def reLU_derivacija(podaci: NDArray) -> NDArray:
    return (podaci > 0).astype(podaci.dtype)

def KL_divergencija(ro : float | NDArray, ro_hat : NDArray) -> NDArray:
    return ro * np.log(ro / ro_hat) + (1 - ro)*np.log((1 - ro)/(1 - ro_hat))

def KL_divergencija_derivacija(ro : float | NDArray, ro_hat : NDArray) -> NDArray:
    return -ro/ro_hat + (1 - ro)/(1 - ro_hat)

class Layer:
    W: NDArray
    b: NDArray
    a: NDArray
    z: NDArray  
    vW: NDArray
    vb: NDArray

    def __init__(self, velic_prosli: int, velic_trenutni: int):
        self.W = 0.01 * np.random.normal(size = (velic_trenutni, velic_prosli))
        self.b = 0.01 * np.random.normal(size = (velic_trenutni,))
        self.a = np.zeros(shape = velic_trenutni)
        self.z = np.zeros(shape = velic_trenutni)
        self.vW = np.zeros_like(self.W)
        self.vb = np.zeros_like(self.b)
    
class SparseAutoencoder:
    slojevi: list[Layer]
    ulaz: NDArray
    greske: list[float]
    aktivacijska: FunctionType
    aktivacijska_derivacija: FunctionType

    def __init__(self, velicine_slojeva: list[int], bottleneck_sloj : int, ro = 0.05, beta = 0.1, momentum = 0.9, aktivacijska_fja = "sigmoid"):
        if aktivacijska_fja == "reLU":
            self.aktivacijska = reLU
            self.aktivacijska_derivacija = reLU_derivacija
        else:
            self.aktivacijska = sigmoid
            self.aktivacijska_derivacija = sigmoid_derivacija
        self.slojevi = []
        self.greske = []
        self.bottleneck_indeks = bottleneck_sloj
        self.ro = ro
        self.beta = beta
        self.momentum = momentum
        for i in range(len(velicine_slojeva) - 1):
            self.slojevi.append(Layer(velicine_slojeva[i], velicine_slojeva[i+1]))

    def forwardPass(self, podaci: NDArray) -> NDArray:
        self.ulaz = podaci
        a = podaci
        for sloj in self.slojevi:
            sloj.z = sloj.W @ a + sloj.b
            sloj.a = self.aktivacijska(sloj.z)
            a = sloj.a
        return a
    
    def backPass(self, y: NDArray, eta: float, ro_hat : NDArray) -> None:
        L = self.slojevi[-1]
        delta = self.aktivacijska_derivacija(L.z) * (L.a - y)

        for i in range(len(self.slojevi) - 2, -1, -1):
            sloj = self.slojevi[i]
            
            W_sljedeci = self.slojevi[i+1].W.copy()

            #self.slojevi[i+1].W -= eta * np.outer(delta, sloj.a)
            #self.slojevi[i+1].b -= eta * delta
            self.slojevi[i+1].vW = self.momentum * self.slojevi[i+1].vW - eta * np.outer(delta, sloj.a)
            self.slojevi[i+1].W += self.slojevi[i+1].vW
            self.slojevi[i+1].vb = self.momentum * self.slojevi[i+1].vb - eta * delta
            self.slojevi[i+1].b += self.slojevi[i+1].vb
            
            if i == self.bottleneck_indeks:
                delta = self.aktivacijska_derivacija(sloj.z) * (W_sljedeci.T @ delta + self.beta * KL_divergencija_derivacija(self.ro, ro_hat))
            else:
                delta = self.aktivacijska_derivacija(sloj.z) * (W_sljedeci.T @ delta)

        #self.slojevi[0].W -= eta * np.outer(delta, self.ulaz)
        #self.slojevi[0].b -= eta * delta
        self.slojevi[0].vW = self.momentum * self.slojevi[0].vW - eta * np.outer(delta, self.ulaz)
        self.slojevi[0].W += self.slojevi[0].vW
        self.slojevi[0].vb = self.momentum * self.slojevi[0].vb - eta * delta
        self.slojevi[0].b += self.slojevi[0].vb
    

    def train(self, podaci: NDArray, oznake: NDArray, n_epochs: int,
              eta: float, opadajuci = False) -> None:
        eta0 = eta

        for epoch in range(n_epochs):
            # Prvo racunam prosjecnu aktivaciju bottleneck sloja
            bottleneck_a = []
            if opadajuci:
                eta = eta0 / (1 + epoch)
            for i in range(len(podaci)):
                self.forwardPass(podaci[i])
                bottleneck_a.append(self.slojevi[self.bottleneck_indeks].a.copy())
            ro_hat = np.mean(bottleneck_a, axis=0)
            ro_hat = np.clip(ro_hat, 1e-8, 1 - 1e-8)

            # Sad radim training loop
            for i in np.random.permutation(len(podaci)):
                self.forwardPass(podaci[i])
                self.greske.append(float(np.linalg.norm(self.slojevi[-1].a - oznake[i])))
                self.backPass(oznake[i], eta, ro_hat)

