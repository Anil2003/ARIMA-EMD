
import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt
%matplotlib inline
plt.rc('font',weight='bold')
legend_properties = {'weight':'bold'}
%matplotlib inline
SMALL_SIZE = 10
MEDIUM_SIZE = 15
BIGGER_SIZE = 20

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
T=D[0:105120]
F=T[0:2000]

from PyEMD import EMD
IMF = EMD().emd(F)
N = IMF.shape[0]+1

# Plot results
plt.figure(figsize=(12,9))
plt.subplot(N,1,1)
plt.plot(F, 'r')
plt.title("Input signal")
plt.xlabel("Time [s]")

for n, imf in enumerate(IMF):
    plt.figure(figsize=(12,9))
    plt.subplot(N,1,n+2)
    plt.plot(imf, 'g')
    plt.title("IMF "+str(n+1))
    plt.xlabel("Time [s]")

IMF[1]

model = pf.ARIMA(data=IMF[1], ar=4, integ=0, ma=4, target='speed', family=pf.Normal())

x = model.fit("MLE")
x.summary()

model.plot_fit(figsize=(15,10))

model.plot_predict_is(h=50, figsize=(10,3))

prediction_a1 = model.predict_is(h=500, fit_once=True)

model = pf.ARIMA(data=IMF[2], ar=4, integ=0, ma=4, target='speed', family=pf.Normal())

x = model.fit("MLE")
x.summary()

model.plot_fit(figsize=(15,10))

model.plot_predict_is(h=50, figsize=(10,3))

prediction_a2 = model.predict_is(h=500, fit_once=True)

model = pf.ARIMA(data=IMF[3], ar=4, integ=0, ma=4, target='speed', family=pf.Normal())

x = model.fit("MLE")
x.summary()

model.plot_fit(figsize=(15,10))

model.plot_predict_is(h=50, figsize=(10,3))

prediction_a3 = model.predict_is(h=500, fit_once=True)

model = pf.ARIMA(data=IMF[4], ar=4, integ=0, ma=4, target='speed', family=pf.Normal())

x = model.fit("MLE")
x.summary()

model.plot_fit(figsize=(15,10))

model.plot_predict_is(h=50, figsize=(10,3))

prediction_a4 = model.predict_is(h=500, fit_once=True)

model = pf.ARIMA(data=IMF[5], ar=4, integ=0, ma=4, target='speed', family=pf.Normal())

x = model.fit("MLE")
x.summary()

model.plot_fit(figsize=(15,10))

model.plot_predict_is(h=50, figsize=(10,3))

prediction_a5 = model.predict_is(h=500, fit_once=True)

model = pf.ARIMA(data=IMF[6], ar=4, integ=0, ma=4, target='speed', family=pf.Normal())

x = model.fit("MLE")
x.summary()

model.plot_fit(figsize=(15,10))

model.plot_predict_is(h=50, figsize=(10,3))

prediction_a6 = model.predict_is(h=500, fit_once=True)

model = pf.ARIMA(data=IMF[7], ar=4, integ=0, ma=4, target='speed', family=pf.Normal())

x = model.fit("MLE")
x.summary()

model.plot_fit(figsize=(15,10))

model.plot_predict_is(h=50, figsize=(10,3))

prediction_a7 = model.predict_is(h=500, fit_once=True)

model = pf.ARIMA(data=IMF[8], ar=4, integ=0, ma=4, target='speed', family=pf.Normal())

x = model.fit("MLE")
x.summary()

model.plot_fit(figsize=(15,10))

model.plot_predict_is(h=50, figsize=(10,3))

prediction_a8 = model.predict_is(h=500, fit_once=True)

model = pf.ARIMA(data=IMF[9], ar=4, integ=0, ma=4, target='speed', family=pf.Normal())

x = model.fit("MLE")
x.summary()

model.plot_fit(figsize=(15,10))

model.plot_predict_is(h=50, figsize=(10,3))

prediction_a9 = model.predict_is(h=500, fit_once=True)

prediction_a=prediction_a1+prediction_a2+prediction_a3+prediction_a4+prediction_a5+prediction_a6+prediction_a7+prediction_a8+prediction_a9

mean_absolute_error(prediction_a,F[-500:])
np.sqrt(mean_squared_error(prediction_a,F[-500:]))
r2_score(prediction_a,F[-500:])

