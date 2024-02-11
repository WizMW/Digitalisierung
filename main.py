from packages.Particleanalyzer import ParticleAnalyzer
import cv2
import os
import numpy as np
# import matplotlib.pyplot as plt
# import packages.ParticleDistribution as dist
# import scipy.stats as stats
# import matplotlib.pyplot as plt
import pandas as pd
from packages.filefinder import list_jpg_files

speeds = np.arange(10, 101, 10)
data_area = pd.DataFrame()

for speed in speeds:
    folder_path = f'C:/Users/Merli/Desktop/python digitalisierung/input_files/{str(speed)}%'
    images = list_jpg_files(folder_path)
    for i, image in enumerate(images):
        analyzer = ParticleAnalyzer(image)
        data_img = analyzer.process_image()
        data_img = data_img.rename(columns={data_img.columns[0]: f"{str(speed)}%_{i}"})
        cv2.imwrite(f"exp/{os.path.basename(image)}", analyzer.contimpgpp)
        data_area = pd.concat([data_area, data_img], axis=1) 
data_area.to_csv('exp/Area.csv', index=False)
# Q0, Q3 = dist.Distribution(data_img, hist_size = 1000)

# Passende Verteilungsfunktion finden

# https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python