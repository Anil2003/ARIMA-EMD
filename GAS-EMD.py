
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

model = pf.GAS(ar=2, sc=2, data=IMF[1], family=pf.Normal())

x = model.fit("MLE")
x.summary()

model.plot_fit(figsize=(15,10))

model.plot_predict_is(h=50, fit_once=True, figsize=(10,3))

prediction_g1 = model.predict_is(h=500, fit_once=True)

model = pf.GAS(ar=2, sc=2, data=IMF[2], family=pf.Normal())

x = model.fit("MLE")
x.summary()

model.plot_fit(figsize=(15,10))

model.plot_predict_is(h=50, fit_once=True, figsize=(10,3))

prediction_g2 = model.predict_is(h=500, fit_once=True)

model = pf.GAS(ar=2, sc=2, data=IMF[3], family=pf.Normal())

x = model.fit("MLE")
x.summary()

model.plot_fit(figsize=(15,10))

model.plot_predict_is(h=50, fit_once=True, figsize=(10,3))

prediction_g3 = model.predict_is(h=500, fit_once=True)

model = pf.GAS(ar=2, sc=2, data=IMF[4], family=pf.Normal())

x = model.fit("MLE")
x.summary()

model.plot_fit(figsize=(15,10))

model.plot_predict_is(h=50, fit_once=True, figsize=(10,3))

prediction_g4 = model.predict_is(h=500, fit_once=True)

model = pf.GAS(ar=2, sc=2, data=IMF[5], family=pf.Normal())

x = model.fit("MLE")
x.summary()

model.plot_fit(figsize=(15,10))

model.plot_predict_is(h=50, fit_once=True, figsize=(10,3))

prediction_g5 = model.predict_is(h=500, fit_once=True)

model = pf.GAS(ar=2, sc=2, data=IMF[6], family=pf.Normal())

x = model.fit("MLE")
x.summary()

model.plot_fit(figsize=(15,10))

model.plot_predict_is(h=50, fit_once=True, figsize=(10,3))

prediction_g6 = model.predict_is(h=500, fit_once=True)

model = pf.GAS(ar=2, sc=2, data=IMF[7], family=pf.Normal())

x = model.fit("MLE")
x.summary()

model.plot_fit(figsize=(15,10))

model.plot_predict_is(h=50, fit_once=True, figsize=(10,3))

prediction_g7 = model.predict_is(h=500, fit_once=True)

model = pf.GAS(ar=2, sc=2, data=IMF[8], family=pf.Normal())

x = model.fit("MLE")
x.summary()

model.plot_fit(figsize=(15,10))

model.plot_predict_is(h=50, fit_once=True, figsize=(10,3))

prediction_g8 = model.predict_is(h=500, fit_once=True)

model = pf.GAS(ar=2, sc=2, data=IMF[9], family=pf.Normal())

x = model.fit("MLE")
x.summary()

model.plot_fit(figsize=(15,10))

model.plot_predict_is(h=50, fit_once=True, figsize=(10,3))

prediction_g9 = model.predict_is(h=500, fit_once=True)

prediction_g=prediction_g1+prediction_g2+prediction_g3+prediction_g4+prediction_g5+prediction_g6+prediction_g7+prediction_g8+prediction_g9

mean_absolute_error(prediction_g,F[-500:])
np.sqrt(mean_squared_error(prediction_g,F[-500:]))
r2_score(prediction_g,F[-500:])

