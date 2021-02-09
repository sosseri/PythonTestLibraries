#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:23:14 2020

@author: alessandroseri
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 11:02:37 2020

@author: alessandroseri
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import io
import requests

# The SIR model differential equations.
def deriv(y, t, N, beta, gamma):
    S, I, R = y
    dSdt = -beta * S * I / N
    dIdt = beta * S * I / N - gamma * I
    dRdt = gamma * I
    return dSdt, dIdt, dRdt

def sir_model(t, N, beta, gamma):
    # Total population, N.
    # Initial number of infected and recovered individuals, I0 and R0.
    I0, R0 = 1, 0
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return I

def expfun(x,a,b,x0):
    return a*np.exp(b*x-x0)

def logistic(x,a,b,c):
    return a/(1+np.exp(b-c*x))
    
def analysis_country(data_death, date_case, country, *time_analysis):
    popt_c=np.zeros([len(country),3])
    popt_d=np.zeros([len(country),3])
    cases_t_days=np.zeros([len(country)])
    deaths_t_days=np.zeros([len(country)])
    t0_c=np.zeros([len(country)])
    t0_d=np.zeros([len(country)])
#    for ii in range(len(country)):
#        print('enter country ' + country[ii])
#        cases = data_case[country[ii]]
#        threashold = np.max(cases)/20
#        cases2fit=np.array(cases)[np.where(cases>threashold)[0]]
#        time_fit_c = np.where(cases>threashold)[0]
#        t0_c[ii]=time_fit_c[0]
#        popt_c[ii,:], pcov_c = curve_fit(expfun, time_fit_c[0:5], cases2fit[0:5])
#        print('fitted cases country ' + str(country[ii]))
#        deaths = data_death[country[ii]]
#        threashold = np.max(deaths)/20
#        deaths2fit=np.array(deaths)[np.where(deaths>0)[0]]
#        time_fit_d = np.where(deaths>0)[0]
#        t0_d[ii]=time_fit_d[0]
#        popt_d[ii,:], pcov_d = curve_fit(expfun, time_fit_d[0:5], deaths2fit[0:5])
#        print('fitted deaths country ' + str(country[ii]))
#        
#        cases_t_days[ii] = expfun(time_fit_c[time_fit_c.argmax()]+time_analysis, *popt_c[ii,:])
#        deaths_t_days[ii] = expfun(time_fit_d[time_fit_d.argmax()]+time_analysis, *popt_d[ii,:])
#        
#    if time_analysis is None:
#        time_analysis = int(0)
        
        
    
    return {'death_time_t0': t0_d, 'cases_time_t0': t0_c, 
            'death_fit':popt_d, 'cases_fit':popt_c,
            'cases_in_t_days':cases_t_days, 'deaths_in_t_days':deaths_t_days}


#%% read csv from url
url_c = 'https://covid.ourworldindata.org/data/ecdc/total_cases.csv'
url_d = 'https://covid.ourworldindata.org/data/ecdc/total_deaths.csv'
s = requests.get(url_c).content
data_case = pd.read_csv(io.StringIO(s.decode('utf-8')))
s = requests.get(url_d).content
data_death = pd.read_csv(io.StringIO(s.decode('utf-8')))

t_analysis = 8 # days
country = ['Italy','Spain','France']
#country = ['Italy']

date_c = data_case['date']
date_d = data_death['date']

data_case = data_case[country]
data_case[np.isnan(data_case)] = 0

data_death = data_death[country]
data_death[np.isnan(data_death)] = 0

data = analysis_country(data_death, data_case,country, t_analysis)
#%%
    
plt.figure(0)
plt.clf()
string_legend='';
for ii in range(len(country)):
    plt.plot(range(len(date_d)),data_death[country[ii]],'o')
plt.legend(country)
for ii in range(len(country)):
    xaxis = np.linspace(data['death_time_t0'][ii],len(date_d),100)
    plt.plot(xaxis,expfun(xaxis,*data['death_fit'][ii,:]),'--k')
plt.legend(country)
#plt.yscale('log')
plt.xlabel('Days starting from ' + date_d[0])
plt.ylabel('Dead people')
plt.xlim([30,xaxis[-1]])
plt.ylim([1,np.max(np.max(data_death))*1.25])
#%%
plt.figure(1)
plt.clf()
string_legend='';
for ii in range(len(country)):
    plt.plot(range(len(date_c)),data_case[country[ii]],'o')
plt.legend(country)
for ii in range(len(country)):
    xaxis = np.linspace(data['cases_time_t0'][ii],len(date_c),100)
    plt.plot(xaxis,expfun(xaxis,*data['cases_fit'][ii,:]),'--k')
plt.legend(country)
#plt.yscale('log')
plt.xlabel('Days starting from ' + date_d[0])
plt.ylabel('Positive people')
plt.xlim([30,xaxis[-1]])
plt.ylim([1,np.max(np.max(data_case))*1.25])

plt.figure(2)
plt.clf()
for ii in range(len(country)):
    plt.plot(range(len(date_c)),data_death[country[ii]]/data_case[country[ii]],'o')
plt.legend(country)
plt.yscale('log')
plt.xlabel('Days starting from ' + date_d[0])
plt.ylabel('Percentage dead people')
plt.xlim([40,xaxis[-1]])
