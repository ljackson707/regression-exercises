from pydataset import data
import pandas as pd 
import numpy as np 
import seaborn as sns

import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, r2_score, explained_variance_score

from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression 
from math import sqrt
import warnings
warnings.filterwarnings('ignore')
from statsmodels.formula.api import ols

def time(df):
    
    x = df['total_bill']
    y = df['tip']
    
    df['baseline'] = df.tip.mean()
    
    ols_model = ols('tip ~ total_bill', data=df).fit()

    df['yhat'] = ols_model.predict(df.total_bill)
    
    df['residual'] = df.tip - df.yhat
    df['baseline_residual'] = df.tip - df.tip.mean()
    
    df['residual^2'] = df.residual**2
    df['baseline_residual^2'] = df.baseline_residual**2 
    
    return df

def plot_residuals(df):

    fig = plt.figure(figsize = (18,5))

    r = plt.subplot(131)
    plt.scatter(df.total_bill, df.residual)
    plt.axhline(y = 0, ls = ':')
    plt.title('OLS model residuals');

    b = plt.subplot(132)
    plt.scatter(df.total_bill, df.baseline_residual)
    plt.axhline(y = 0, ls = ':')
    plt.title('Baseline Residuals');

def regression_errors(df):
    
    SSE = df['residual^2'].sum()
    
    TSS = df['baseline_residual^2'].sum()

    MSE = SSE/len(df)
 
    RMSE = sqrt(MSE)
    
    return SSE, TSS, MSE, RMSE

def baseline_mean_errors(df):
    
    SSE_baseline = df['baseline_residual^2'].sum()

    MSE_baseline = SSE_baseline/len(df)
 
    RMSE_baseline = sqrt(MSE_baseline)
    
    return SSE_baseline, MSE_baseline, RMSE_baseline

def model_significance(df):
    ols_model = ols('tip ~ total_bill', data=df).fit()

    df['yhat'] = ols_model.predict(df.total_bill)
    return ols_model.summary()

#~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
import math
import sklearn.metrics
import pandas as pd
import numpy as  np
import matplotlib.pyplot as plt

def residuals(actual, predicted):
    return actual - predicted

def sse(actual, predicted):
    return (residuals(actual, predicted) **2).sum()

def mse(actual, predicted):
    n = actual.shape[0]
    return sse(actual, predicted) / n

def rmse(actual, predicted):
    return math.sqrt(mse(actual, predicted))

def ess(actual, predicted):
    return ((predicted - actual.mean()) ** 2).sum()

def tss(actual):
    return ((actual - actual.mean()) ** 2).sum()

def regression_errors(actual, predicted):
    return pd.Series({
        'sse': sse(actual, predicted),
        'ess': ess(actual, predicted),
        'tss': tss(actual),
        'mse': mse(actual, predicted),
        'rmse': rmse(actual, predicted),
    })

def baseline_mean_errors(actual):
    predicted = actual.mean()
    return {
        'sse': sse(actual, predicted),
        'mse': mse(actual, predicted),
        'rmse': rmse(actual, predicted),
    }

def better_than_baseline(actual, predicted):
    rmse_baseline = rmse(actual, actual.mean())
    rmse_model = rmse(actual, predicted)
    return rmse_model < rmse_baseline

def model_significance(ols_model):
    return {
        'r^2 -- variance explained': ols_model.rsquared,
        'p-value -- P(data|model == baseline)': ols_model.f_pvalue,
    }
def plot_residuals(actual, predicted):
    residuals = actual - predicted
    plt.hlines(0, actual.min(), actual.max(), ls=':')
    plt.scatter(actual, residuals)
    plt.ylabel('residual ($y - \hat{y}$)')
    plt.xlabel('actual value ($y$)')
    plt.title('Actual vs Residual')
    plt.show()