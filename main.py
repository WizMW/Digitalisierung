from packages.Particleanalyzer import ParticleAnalyzer
import cv2
import os
import numpy as np
#import matplotlib.pyplot as plt
import packages.ParticleDistribution as dist
import scipy.stats as stats
import matplotlib.pyplot as plt
import pandas as pd

image_list = [
    r'C:\Users\Merli\Desktop\python digitalisierung\input_files\10%\IMG_0011.JPG',
    r'C:\Users\Merli\Desktop\python digitalisierung\input_files\20%\IMG_0021.JPG',
    r'C:\Users\Merli\Desktop\python digitalisierung\input_files\30%\IMG_0031.JPG',
    r'C:\Users\Merli\Desktop\python digitalisierung\input_files\40%\IMG_0041.JPG',
    r'C:\Users\Merli\Desktop\python digitalisierung\input_files\50%\IMG_0051.JPG',
    r'C:\Users\Merli\Desktop\python digitalisierung\input_files\60%\IMG_0061.JPG',
    r'C:\Users\Merli\Desktop\python digitalisierung\input_files\70%\IMG_0073.JPG',
    r'C:\Users\Merli\Desktop\python digitalisierung\input_files\80%\IMG_0081.JPG',
    r'C:\Users\Merli\Desktop\python digitalisierung\input_files\90%\IMG_0091.JPG',
    r'C:\Users\Merli\Desktop\python digitalisierung\input_files\100%\IMG_0101.JPG',

]


for image in image_list:

    analyzer = ParticleAnalyzer(image)
    data_img = analyzer.process_image()
    # cv2.imwrite("exp/" + os.path.basename(image), analyzer.contimpgpp)
    Q0, Q3 = dist.Distribution(data_img, hist_size = 1000)

    # Passende Verteilungsfunktion finden

    # https://stackoverflow.com/questions/6620471/fitting-empirical-distribution-to-theoretical-ones-with-scipy-python