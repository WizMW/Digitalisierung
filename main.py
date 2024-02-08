from packages.Particleanalyzer import ParticleAnalyzer
import cv2
import os
import numpy as np
#import matplotlib.pyplot as plt

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
    # lenght of a sphere
    data_img = data_img.apply(lambda x: np.sqrt(x * 4/np.pi))
    # Volume of spheres
    data_img = data_img.apply(lambda x: 4/3*((x/2)**3) * np.pi)
    # print(data_img.mean())
    cv2.imwrite("exp/" + os.path.basename(image), analyzer.contimpgpp)
    hist_value, _ = np.histogram(data_img, bins=100)
    Q3 = np.cumsum(hist_value)
    Q3 = Q3/np.max(Q3)
    print(Q3)
    np.save("exp/" + (os.path.basename(image)).replace(".JPG", "") + ".npy", Q3)
