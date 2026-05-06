from dataclasses import dataclass
from typing import List

@dataclass
class Config:
    n_razina: int = 3
    uzorkovanje_postotci: List[float] = None
    velicine_patcha: List[int] = None  #po razini
    W_po_razini: List[int] = None
    M_po_razini: List[int] = None
    k: int = 16
    d: int = 6
    t: int = 1
    r: float = 20.0
    koristiti_laplacian: bool = True
    def __post_init__(self):
        if self.uzorkovanje_postotci is None:
            self.uzorkovanje_postotci = [0.10, 0.33, 0.50, 0.50] 
        if self.velicine_patcha is None:
            self.velicine_patcha = [2, 2, 4, 8]  #od najgrubljeg do najfinijeg
        if self.W_po_razini is None:
            self.W_po_razini = [13, 13, 10, 20]
        if self.M_po_razini is None:
            self.M_po_razini = [5, 5, 2, 4]