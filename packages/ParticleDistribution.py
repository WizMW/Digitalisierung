import numpy as np
import pandas as pd
from scipy.stats import weibull_min
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt


class Distribution:
    def __init__(self, data_img):
        self.q0 = None
        self.Q0 = None
        self.q3 = None
        self.Q3 = None
        self.data_area = data_img.dropna().to_numpy()
        self.bins = None 

    def PGV(self, hist_size = 1000):
        diameter = np.sqrt(self.data_area/np.pi)
        self.q0, xi = np.histogram(diameter, hist_size)
        self.Q0 = np.cumsum(self.q0) / np.sum(self.q0)
        # Q3 Berechnung
        x_mean = (xi + np.roll(xi,-1)) / 2
        x_mean = x_mean[:-1]
        self.q3 = ((x_mean**3) * self.q0)/np.sum((x_mean**3) * self.q0 * (xi[1]-xi[0]))
        self.Q3 = np.cumsum(self.q3) / np.sum(self.q3)
        # Histogramm plotten
        self.bins = np.linspace(0, np.max(diameter), len(self.q3))

    def fit_weibull(self):
        # Define the Weibull cumulative distribution function
        def weibull_cdf(x, c, loc, scale):
            return weibull_min.cdf(x, c, loc=loc, scale=scale)

        self.PGV()
        # Define the cumulative probabilities
        x = self.bins
        # Curve fitting using Least Squares Method (LSM)
        params, _ = curve_fit(weibull_cdf, x, self.Q3, p0 = [1, 0, 1])
        # Extract the estimated parameters
        shape, loc, scale = params
        # Return the estimated parameters
        # plt.plot(self.bins, self.Q3, color='blue',label='$Q_3$ Verteilung')
        # plt.plot(self.bins, weibull_min.cdf(x, shape, loc=loc, scale=scale), color='red', label='Weibull-Fit')
        # plt.legend()
        # plt.xlabel('Durchmesser (mm)',fontsize=20)
        # plt.ylabel('Massenanteil ($m_i$/m)',fontsize=20)
        # plt.title('$Q_3$ Verteilung',fontsize=25)
        # plt.grid(True)
        # plt.show()




        return shape, loc, scale

df = pd.read_csv('exp\Area.csv')












# data_here = df['10%_0']
# Dist = Distribution(data_here)
# Dist.PGV()




column_names = df.columns.tolist()
Weibull_data = pd.DataFrame()
speeds = np.arange(10, 101, 10)

for speed in speeds:
    i = 0
    while f'{str(speed)}%_{str(i)}' in df.columns:
        Dist = Distribution(df[f'{str(speed)}%_{str(i)}'])
        shape, loc, scale = Dist.fit_weibull()
        index = len(Weibull_data)
        Weibull_data.loc[index, 'shape'] = shape
        Weibull_data.loc[index, 'loc'] = loc
        Weibull_data.loc[index, 'scale'] = scale
        i += 1

Weibull_data["Names"] = column_names
Weibull_data = Weibull_data.set_index('Names')
Weibull_data.to_csv('Weibull_dist.csv')
