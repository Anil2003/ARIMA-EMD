
from numpy import *
from pylab import plot, show
import nolds
import numpy as np
import pandas as pd
import pyflux as pf
from datetime import datetime
import matplotlib.pyplot as plt
%matplotlib inline

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

data = pd.read_csv('dataset-2007.csv', usecols=['wind direction at 100m (deg)', 'wind speed at 100m (m/s)', 'air temperature at 2m (K)', 'surface air pressure (Pa)', 'density at hub height (kg/m^3)'], skiprows=3)
data.columns = ['direction', 'speed', 'temp', 'pressure', 'density']
data.head(3)

D=data['speed'].values
T=D[0:105120:12]
F=T[0:2000]

h = nolds.dfa(F)
h

RS=nolds.hurst_rs(F)
RS

# calculate standard deviation of differenced series using various lags
lags = range(2, 20)
tau = [sqrt(std(subtract(F[lag:], F[:-lag]))) for lag in lags]

# plot on log-log scale
plot(log(lags), log(tau)); show()

# calculate Hurst as slope of log-log plot
m = polyfit(log(lags), log(tau), 1)
hurst = m[0]*2.0
hurst

#farctal dimension (correlation dimension)= slope of the line fitted to log(r) vs log(C(r))
# If the correlation dimension is constant for all ‘m’ the time series will be deterministic
#if the correlation exponentincreases with increase in ‘m’ the time series will be stochastic.
h01 = nolds.corr_dim(F,2,debug_plot=True)
h01

#lyap_r = estimate largest lyapunov exponent
h1=nolds.lyap_r(F,emb_dim=2,debug_plot=True)
h1

#lyap_e = estimate whole spectrum of lyapunov exponents
h2=nolds.lyap_e(F)
h2

from pyentrp import entropy as ent
T1=np.std(F)
T1
k= 0.2*T1
k

#sample entropy
h = nolds.sampen(F,3,tolerance=k)
h

#permutation entropy
h2=ent.permutation_entropy(F,order=3,normalize=True)
h2
