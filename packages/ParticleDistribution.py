import numpy as np
import pandas as pd
from scipy.stats import weibull_min
from scipy.optimize import curve_fit


class Distribution:
    def __init__(self, data_img):
        self.q3 = None
        self.Q0 = None
        self.q3 = None
        self.Q3 = None
        self.data_area = data_img.dropna().to_numpy()

    def PVG(self, hist_size = 1000):
        diameter = np.sqrt(self.data_area * 4/np.pi)
        self.q0, _ = np.histogram(diameter, hist_size)
        self.Q0 = np.cumsum(self.q0) / np.sum(self.q0)
        mass = 4/3 * np.pi * np.square(diameter/2)
        self.q3, _ = np.histogram(mass, hist_size)
        self.Q3 = np.cumsum(self.q3) / np.sum(self.q3)
        return self.Q3


    def fit_weibull(self):
        if self.Q3 is None:
            self.PVG()
        # Define the cumulative probabilities
        x = np.arange(len(self.Q3))

        # Define the Weibull cumulative distribution function
        def weibull_cdf(x, c, loc, scale):
            return weibull_min.cdf(x, c, loc=loc, scale=scale)

        # Curve fitting using Least Squares Method (LSM)
        params, _ = curve_fit(weibull_cdf, x, self.Q3)
        # Extract the estimated parameters
        shape, loc, scale = params
        # Return the estimated parameters
        return shape, loc, scale

df = pd.read_csv('exp\Area.csv')

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
