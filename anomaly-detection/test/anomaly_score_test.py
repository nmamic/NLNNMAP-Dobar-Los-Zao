import os
import sys
from pathlib import Path
import numpy as np
from PIL import Image

SRC_PATH = Path(os.path.dirname(Path.cwd())) / "anomaly-detection/src"
sys.path.insert(0, str(SRC_PATH))

from config import Config
from piramida import run_pyramid

def ucitaj_sliku(putanja, ciljana_velicina=(200, 200)):
    slika = Image.open(putanja)
    
    #pretvori u grayscale ako nije
    if slika.mode != 'L':
        slika = slika.convert('L')
    
    #smanji na 200x200 kao u radu
    slika = slika.resize(ciljana_velicina, Image.LANCZOS)
    
    #pretvori u numpy i normaliziraj
    slika_array = np.array(slika, dtype=np.float64) / 255.0
    
    return slika_array

#Ships dataset
"""
images_path = Path.cwd() / "data/ships"
images = [i for i in os.listdir(images_path)
          if not i.endswith("true.jpg")]

#Laplacian - Anomaly Score
config = Config(koristiti_laplacian=True, r=10)
for i in images:
    slika = ucitaj_sliku(images_path / i, (200, 200))
    anomaly_score= run_pyramid(slika, config)
    np.save("test/output/ships/lap_anom_" + i.removesuffix(".jpg"),
            anomaly_score)

#Laplacian - Saliency Score
config = Config(koristiti_laplacian=True,
                koristiti_saliency_score=True,
                r=10)
for i in images:
    slika = ucitaj_sliku(images_path / i, (200, 200))
    anomaly_score= run_pyramid(slika, config)
    np.save("test/output/ships/lap_sal_" + i.removesuffix(".jpg"),
            anomaly_score)
"""

#Shipwrecks dataset
"""
images_path = Path.cwd() / "data/shipwrecks"
images = [i for i in os.listdir(images_path)
          if i.endswith("500.jpg") and not i.startswith("Art")]


#Laplacian - Anomaly Score
config = Config(koristiti_laplacian=True, uzorkovanje_postotci = [0.50, 0.35, 0.20])
for i in images:
    slika = ucitaj_sliku(images_path / i, (200, 200))
    anomaly_score= run_pyramid(slika, config)
    np.save("test/output/shipwrecks/lap_anom_" + i.removesuffix(".jpg"),
            anomaly_score)

               
#Nystrom - Anomaly Score
config = Config(koristiti_laplacian=False, uzorkovanje_postotci = [0.50, 0.35, 0.20])
for i in images:
    slika = ucitaj_sliku(images_path / i, (200, 200))
    anomaly_score = run_pyramid(slika, config)
    np.save("test/output/shipwrecks/nys_anom_" + i.removesuffix(".jpg"),
               anomaly_score)

#Laplacian - Saliency Score
config = Config(koristiti_laplacian=True,
                koristiti_saliency_score=True)
for i in images:
    slika = ucitaj_sliku(images_path / i, (200, 200))
    anomaly_score= run_pyramid(slika, config)
    np.save("test/output/shipwrecks/lap_sal_" + i.removesuffix(".jpg"),
            anomaly_score)

               
#Nystrom - Saliency Score
config = Config(koristiti_laplacian=False,
                koristiti_saliency_score=True)
for i in images:
    slika = ucitaj_sliku(images_path / i, (200, 200))
    anomaly_score = run_pyramid(slika, config)
    np.save("test/output/shipwrecks/nys_sal_" + i.removesuffix(".jpg"),
               anomaly_score)


slika = ucitaj_sliku(images_path / "Artificial_Reef_03_500.jpg", (200, 200))
anomaly_score = run_pyramid(slika, Config())
saliency_score = run_pyramid(slika, Config(koristiti_saliency_score=True))
np.save("test/output/shipwrecks/lap_anom_" + "Artificial_Reef_03_500",
        anomaly_score)
np.save("test/output/shipwrecks/lap_sal_" + "Artificial_Reef_03_500",
        saliency_score)
"""


#Mines dataset
images_path = Path.cwd() / "data/mines"
images = [i for i in os.listdir(images_path)
          if i.endswith("2021.jpg")]


#Laplacian - Anomaly Score
config = Config(koristiti_laplacian=True)
for i in images:
    slika = ucitaj_sliku(images_path / i, (168, 168))
    anomaly_score= run_pyramid(slika, config)
    np.save("test/output/mines/lap_anom_" + i.removesuffix(".jpg"),
            anomaly_score)

               
#Nystrom - Anomaly Score
config = Config(koristiti_laplacian=False)
for i in images:
    slika = ucitaj_sliku(images_path / i, (168, 168))
    anomaly_score = run_pyramid(slika, config)
    np.save("test/output/mines/nys_anom_" + i.removesuffix(".jpg"),
               anomaly_score)

#Laplacian - Saliency Score
config = Config(koristiti_laplacian=True,
                koristiti_saliency_score=True)
for i in images:
    slika = ucitaj_sliku(images_path / i, (168, 168))
    anomaly_score= run_pyramid(slika, config)
    np.save("test/output/mines/lap_sal_" + i.removesuffix(".jpg"),
            anomaly_score)

               
#Nystrom - Saliency Score
config = Config(koristiti_laplacian=False,
                koristiti_saliency_score=True)
for i in images:
    slika = ucitaj_sliku(images_path / i, (168, 168))
    anomaly_score = run_pyramid(slika, config)
    np.save("test/output/mines/nys_sal_" + i.removesuffix(".jpg"),
               anomaly_score)
