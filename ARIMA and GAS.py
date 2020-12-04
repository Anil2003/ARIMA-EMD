
import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt
%matplotlib inline
plt.rc('font',weight='bold')
legend_properties = {'weight':'bold'}
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt
%matplotlib inline
SMALL_SIZE = 15
MEDIUM_SIZE = 20
BIGGER_SIZE = 30

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv('dataset-2007.csv', usecols=['wind direction at 100m (deg)', 'wind speed at 100m (m/s)', 'air temperature at 2m (K)', 'surface air pressure (Pa)', 'density at hub height (kg/m^3)'], skiprows=3)
data.columns = ['direction', 'speed', 'temp', 'pressure', 'density']
data.head(3)

D=data['speed'].values
T=D[0:105120:12]
F=T[0:2000]

model = pf.ARIMA(data=F, ar=4, integ=0, ma=4, target='speed', family=pf.Normal())

x = model.fit("MLE")
x.summary()

model.plot_fit(figsize=(15,10))

model.plot_predict_is(h=50, figsize=(10,3))

prediction = model.predict_is(h=500, fit_once=True)

mean_absolute_error(prediction, F[-500:])
np.sqrt(mean_squared_error(prediction, F[-500:]))
r2_score(prediction, F[-500:])


model = pf.GAS(ar=2, sc=2, data=F, family=pf.Normal())
x = model.fit("MLE")
x.summary()
model.plot_fit(figsize=(15,10))

model.plot_predict_is(h=50, fit_once=True, figsize=(10,3))

prediction_g = model.predict_is(h=500, fit_once=True)

mean_absolute_error(prediction_g, F[-500:])
np.sqrt(mean_squared_error(prediction_g, F[-500:]))
r2_score(prediction_g, F[-500:])
