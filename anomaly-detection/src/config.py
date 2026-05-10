from dataclasses import dataclass
from typing import List
@dataclass
class Config:
    n_razina: int = 2
    uzorkovanje_postotci: List[float] = None
    velicine_patcha: List[int] = None
    W_po_razini: List[int] = None
    M_po_razini: List[int] = None
    d_po_razini: List[int] = None  # dodaj dimenziju po razini
    k: int = 16
    t: int = 1
    r: float = 20.0
    koristiti_laplacian: bool = True
    koristiti_saliency_score: bool = False
    saliency_K: int = 64
    saliency_c: float = 3.0

    def __post_init__(self):
        if self.uzorkovanje_postotci is None:
            self.uzorkovanje_postotci = [0.50, 0.33, 0.10]
        if self.velicine_patcha is None:
            self.velicine_patcha = [2, 4, 8]
        if self.W_po_razini is None:
            self.W_po_razini = [6, 10, 20]
        if self.M_po_razini is None:
            self.M_po_razini = [2, 2, 4]
        if self.d_po_razini is None:
            self.d_po_razini = [3, 6, 6]  
