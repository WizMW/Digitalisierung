import numpy as np


class Distribution:
    def __init__(self, data_img):
        self.Q0 = None
        self.Q3 = None
        self.data_area = data_img.values



def Distribution(data_img,hist_size = 1000):
    data_img = data_img.apply(lambda x: np.sqrt(x * 4/np.pi))
    hist_value, _ = np.histogram(data_img, bins=hist_size)
    Q0 = np.cumsum(hist_value) / np.sum(hist_value)
    data_img = data_img.apply(lambda x: 4/3*((x/2)**3) * np.pi)
    Q3 = np.cumsum(hist_value) / np.sum(hist_value)
    return Q0, Q3

Distribution(data)
