import numpy as np
import matplotlib.pyplot as plt
from config import Config
from piramida import run_pyramid


#def stvori_sintetsku_sliku(H=200, W=200, seed=42):
#    np.random.seed(seed)
#    
#    x = np.linspace(0, 4 * np.pi, W)
#    y = np.linspace(0, 4 * np.pi, H)
#    xx, yy = np.meshgrid(x, y)
#    pozadina = np.sin(xx) * np.cos(yy)
#    pozadina += 0.3 * np.random.randn(H, W)
#    pozadina = (pozadina - pozadina.min()) / (pozadina.max() - pozadina.min())
#    
#    image = pozadina.copy()
#    image[90:93, 95:110] = 0.95
#    image[93:96, 95:110] = 0.05
#    
#    return image

def stvori_sintetsku_sliku(H=200, W=200, seed=42):
    np.random.seed(seed)
    
    # uniformna buka u pozadini
    pozadina = np.random.randn(H, W)
    pozadina = (pozadina - pozadina.min()) / (pozadina.max() - pozadina.min())
    
    image = pozadina.copy()
    #sintetski napravim anomaliju, kkao u radu veličine otprilike 3x15
    image[90:93, 95:110] = 0.95
    image[93:96, 95:110] = 0.05
    
    return image


image = stvori_sintetsku_sliku()
config = Config(koristiti_laplacian=True)
scores = run_pyramid(image, config)


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].imshow(image, cmap='gray')
axes[0].set_title('Sintetska sonar slika')
axes[0].axhline(90, color='r', linewidth=0.5)
axes[0].axhline(96, color='r', linewidth=0.5)
axes[0].axvline(95, color='r', linewidth=0.5)
axes[0].axvline(110, color='r', linewidth=0.5)

axes[1].imshow(scores, cmap='hot')
axes[1].set_title('Anomaly score-ovi')
axes[1].colorbar = plt.colorbar(axes[1].images[0], ax=axes[1])

plt.tight_layout()
plt.savefig('test_rezultat.png')
plt.show()

print(f"Max anomaly vrijednost: {scores.max():.3f}")
print(f"Medijan anomaly vrijednost: {scores.mean():.3f}")
print(f"Score u anomalnom području: {scores[90:96, 95:110].mean():.3f}")
print(f"Score u pozadini: {scores[0:50, 0:50].mean():.3f}")
