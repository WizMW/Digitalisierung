import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

class ParticleAnalyzer:
    def __init__(self, image_path, output_folder='Results', thresh_size=[8, 200], size2pixel=0.00581395):
        self.image_path = image_path
        self.thresh_size = thresh_size
        self.size2pixel = size2pixel
        self.img = None
        self.roi = None
        self.binary_image = None
        self.contimg = None
        self.contimpgpp = None


    def image_croping(self):
        self.img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
        start_x, end_x = int(0.25 * self.img.shape[1]), int(0.75 * self.img.shape[1])
        start_y, end_y = int(0 * self.img.shape[0]), int(0.5 * self.img.shape[0])
        self.roi = self.img[start_y:end_y, start_x:end_x]
        return self.roi
    
    def binarization(self):
        # Otsu binarisation
        _, self.binary_image = cv2.threshold(self.roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return self.binary_image
    

    def process_image(self):
        self.image_croping()
        self.binarization()
        contours, _ = cv2.findContours(self.binary_image, cv2.RETR_LIST , cv2.CHAIN_APPROX_SIMPLE )
        self.contimg = (self.roi).copy()
        self.contimpgpp = self.contimg.copy()
        cv2.drawContours(self.contimg, contours, -1, (0, 255, 0), 2)        
        new_contours = ()
        for i, array_3d in enumerate(contours):
            points = array_3d.reshape(-1, 2)

            if not np.isin(0, points).any() or np.isin(self.binary_image.shape[0], points[:,0]).any() or np.isin(self.binary_image.shape[1], points[:,1]):
                new_array = points
                new_array.reshape(array_3d.shape)
                new_contours += (new_array,)

        # Liste zur Speicherung der Radien
        radii_array = np.array([])
        Area_array = np.array([])

        for contour in new_contours:
            if contour.shape[0] >= 5:
                # Fit an ellipse to the contour
                ellipse = cv2.fitEllipse(contour)
                center, axes, angle = ellipse
                                                        
                if not any(math.isnan(value) for value in axes) and max(axes) < self.thresh_size[1] and min(axes) > self.thresh_size[0]:
                    cv2.drawContours(self.contimpgpp, [contour], -1, (0, 255, 0), 2)
                    Area = cv2.contourArea(contour)
                    Area_array = np.append(Area_array,Area)
                    major_axis = max(axes)
                    minor_axis = min(axes)
                    radius = major_axis / 2.0
                    radii_array = np.append(radii_array, radius)
        
        data_img = pd.DataFrame(Area_array, columns=["Particle_Area"])*(self.size2pixel**2)
     #   data_img = data_img.rename(columns={image_path: 'Neuer_Name'})
        return data_img






#from tkinter import Tk
#from tkinter.filedialog import askopenfilename

# Tkinter-Fenster initialisieren
#root = Tk()
#root.withdraw()  # Verhindert, dass das leere Hauptfenster angezeigt wird


#image_path = askopenfilename()


#analyzer = ParticleAnalyzer(image_path)
#data_img = analyzer.process_image()

#cv2.imwrite('1. img.jpg', analyzer.img)
#cv2.imwrite('2. roi.jpg', analyzer.roi)
#cv2.imwrite('3. binary.jpg', analyzer.binary_image)
#cv2.imwrite('4. contour.jpg', analyzer.contimg)
#cv2.imwrite('5. contour_postprocess.jpg', analyzer.contimpgpp)
#data_img.to_csv('60.csv', index=False)
