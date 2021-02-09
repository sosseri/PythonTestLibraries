#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:23:14 2020
LAST MODIFIED 7 MAY
@author: alessandroseri
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import io
import requests
from scipy.integrate import odeint


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
    for ii in range(len(country)):
        print('enter country ' + country[ii])
        cases = data_case[country[ii]]
        # Defining Splits
        splits = 1
        # Finding average of each consecutive segment
        cases = [sum(cases[i:i + splits])/splits for i in range(len(cases) - splits + 1)]
        cases = np.array(cases,dtype='int')
        threashold = 0#np.max(cases)/100
        cases2fit=np.array(cases)[np.where(cases>threashold)[0]]
        time_fit_c = np.where(cases>threashold)[0]
        t0_c[ii]=time_fit_c[0]
        p0 = [np.max(cases)*20,0.22,1/12]
        popt_c[ii,:], pcov_c = curve_fit(sir_model, time_fit_c, cases2fit, p0, bounds=([0,0,0],[np.inf,np.inf,np.inf]))
        print('fitted cases country ' + str(country[ii]))
        deaths = data_death[country[ii]]
        deaths = [sum(deaths[i:i + splits])/splits for i in range(len(deaths) - splits + 1)] 
        deaths = np.array(deaths,dtype='int')
#        threashold = 4#np.max(deaths/100)
        deaths2fit=np.array(deaths)[np.where(deaths>threashold)[0]]
        time_fit_d = np.where(deaths>threashold)[0]
        t0_d[ii]=time_fit_d[0]
        p0 = [np.max(deaths)*15,0.3,1/24]
        popt_d[ii,:], pcov_d = curve_fit(sir_model, time_fit_d, deaths2fit,p0, bounds=([0,0,0],[np.inf,np.inf,np.inf]))
        print('fitted deaths country ' + str(country[ii]))

        cases_t_days[ii] = sir_model(time_fit_c[time_fit_c.argmax()]+time_analysis, *popt_c[ii,:])
        deaths_t_days[ii] = sir_model(time_fit_d[time_fit_d.argmax()]+time_analysis, *popt_d[ii,:])

    if time_analysis is None:
        time_analysis = int(0)


    return {'death_time_t0': t0_d, 'cases_time_t0': t0_c,
            'death_fit':popt_d, 'cases_fit':popt_c,
            'cases_in_t_days':cases_t_days, 'deaths_in_t_days':deaths_t_days}


#%% read csv from url
url_c = 'https://covid.ourworldindata.org/data/ecdc/new_cases.csv'
url_d = 'https://covid.ourworldindata.org/data/ecdc/new_deaths.csv'
s = requests.get(url_c).content
data_case = pd.read_csv(io.StringIO(s.decode('utf-8')))
s = requests.get(url_d).content
data_death = pd.read_csv(io.StringIO(s.decode('utf-8')))

t_analysis = 2 # days
#country = ['Germany','Portugal', 'France']
country = 'Italy'

last_date_analyzed = '2020-05-10';
last_date_indx = np.where(data_case['date']==last_date_analyzed)[0][0]
data_case = data_case[0:last_date_indx]
data_death = data_death[0:last_date_indx]
date_c = data_case['date']
date_d = data_death['date']
date_d = date_d
date_c = date_c

data_case = data_case[country]
data_case[np.isnan(data_case)] = 0

data_death = data_death[country]
data_death[np.isnan(data_death)] = 0
#%%
legend=[]

splits = 2
# Finding average of each consecutive segment
data_case = [sum(data_case[i:i + splits])/splits for i in range(len(data_case) - splits + 1)]
data_case = np.array(data_case,dtype='int')

#data = analysis_country(data_death, data_case,country, t_analysis)
#%% For Italy
if country=='Italy':
    important_dates=[]
    important_dates.append(date_c[np.where(data_case>0)[0][0]])#inizio
#    important_dates.append('2020-02-24')# chiusura 11 comuni al nord
    important_dates.append('2020-03-04')# chiusura scuole e università
    important_dates.append('2020-03-11')# tutta italia zona rossa
    important_dates.append(date_c[len(date_c)-11])# 6 giorni fà
    population = 6e7
##%% for Spain
#if country=='Spain':
#    important_dates=[]
#    important_dates.append(date_c[np.where(data_case>splits-1)[0][0]])#inizio
#    important_dates.append('2020-03-16')# toda Espana emergencia
#    important_dates.append('2020-04-13')# Riapertura lenta 
#    important_dates.append('2020-04-24')# Bambini in giro
#    important_dates.append('2020-05-02')# Tutti in giro
#
#    important_dates.append(date_c[len(date_c)-11])# n giorni fà
#    population = 4.7e7
#%%
date_important_dates = np.zeros(len(important_dates))

for ii in range(len(important_dates)):
    date_important_dates[ii] = np.where(date_c == important_dates[ii])[0]

indx_new = np.arange(date_important_dates[0],len(data_case),dtype='int')
data_case_new = data_case[indx_new]
data_case_new = np.array(data_case_new)
date_important_dates_new = date_important_dates-date_important_dates[0]

def anal_sir_model(data_case, date_start, date_end):
    ds = 0
    de = int(date_end)+4 #4 instead of 7 because there is split 2 (see line127)
    cases = data_case[ds:de]
    
    time = np.linspace(ds,de,de-ds)
    time_fit = np.linspace(ds,len(data_case)+5,len(data_case)-ds+5)
    
    p0 = [population*.9,0.22,1/24]
    popt, pcov = curve_fit(sir_model, time, cases, p0, bounds=([0,0,0.025],[.95*population,np.inf,0.2]))
    print('fitting values ' + str(popt))
    plt.figure(123)
#    plt.plot(time,cases)
    plt.plot(time_fit,sir_model(time_fit,*popt))
    return {'t0': date_start, 'tf': date_end,
            'fit':popt, 'fit_error':pcov,
            'cases':cases, 'time':time}

plt.figure(123)
plt.clf()
plt.plot(data_case_new,'o')
legend+=['Data new cases']
for ii in range(len(date_important_dates)-1):
    a = anal_sir_model(data_case_new, date_important_dates_new[ii], date_important_dates_new[ii+1])
    legend+=['Predictions with data until '+ str(important_dates[ii+1]) +' + 1 week']

for ii in range(len(date_important_dates_new)-2):
    legend+=['Restrictions day '+ important_dates[ii+1]]
    plt.plot(np.array([date_important_dates_new[ii+1], date_important_dates_new[ii+1]]),np.array([0,np.max(data_case_new)+1000]),'--')
plt.ylim([0,np.max(data_case_new)+200])
plt.legend(legend)
plt.xlabel('Days from ' + str(important_dates[0]))
plt.ylabel('Number new cases per day')