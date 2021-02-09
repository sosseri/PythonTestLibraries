#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 12:23:14 2020

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
country = ['Italy','Spain','Germany']

date_c = data_case['date']
date_d = data_death['date']

data_case = data_case[country]
data_case[np.isnan(data_case)] = 0

data_death = data_death[country]
data_death[np.isnan(data_death)] = 0

data = analysis_country(data_death, data_case,country, t_analysis)

#%%
starting_date = 50
plt.figure(0)
plt.clf()
string_legend='';
for ii in range(len(country)):
    starting_plot_d =int(data['death_time_t0'][ii])*0 +starting_date
    plt.plot(range(len(date_d)-1-starting_date),data_death[country[ii]][starting_plot_d:-1],'o')
plt.legend(country)
for ii in range(len(country)):
    starting_plot_d =int(data['death_time_t0'][ii])*0+starting_date
    xaxis = np.linspace(data['death_time_t0'][ii],len(date_d)+20,100)
    plt.plot(xaxis-starting_plot_d,sir_model(xaxis,*data['death_fit'][ii,:]),'--k')
plt.legend(country)
#plt.yscale('log')
#plt.xlabel('Days starting from ' + date_d[0])
#plt.ylabel('Dead people')
plt.xlabel('Giorni a partire dal ' + date_d[starting_date])
plt.ylabel('Nuove morti giornaliere')
#plt.xlim([30,xaxis[-1]])
plt.ylim([1,np.max(np.max(data_death))*1.25])
#%%
plt.figure(1)
plt.clf()
starting_date = 50
string_legend='';
for ii in range(len(country)):
    starting_plot_c = int(data['cases_time_t0'][ii])*0+starting_date
    plt.plot(range(len(date_c)-1-starting_date),data_case[country[ii]][starting_plot_c:-1],'o')
plt.legend(country)
for ii in range(len(country)):
    starting_plot_c = int(data['cases_time_t0'][ii])*0
    xaxis = np.linspace(data['cases_time_t0'][ii],len(date_c)+20,100)
    plt.plot(xaxis-starting_date,sir_model(xaxis,*data['cases_fit'][ii,:]),'--k')
plt.legend(country)
#plt.yscale('log')
#plt.xlabel('Days starting from ' + date_d[0])
#plt.ylabel('Positive people')
plt.xlabel('Giorni a partire dal ' + date_d[starting_date])
plt.ylabel('Nuovi positivi giornalieri')
#plt.xlim([30,xaxis[-1]])
plt.ylim([1,np.max(np.max(data_case))*1.25])

#%%
plt.figure(2)
plt.clf()
for ii in range(len(country)):
    plt.plot(range(len(date_c)),data_death[country[ii]]/data_case[country[ii]],'o')
plt.legend(country)
plt.yscale('log')
plt.xlabel('Days starting from ' + date_d[0])
plt.ylabel('Percentage dead people')
plt.xlim([40,xaxis[-1]])
#%%
for ii in range(len(country)):
    print('The expected cases in '+country[ii]+' in '+ str(t_analysis) + ' days are ' +  str(int(data['cases_in_t_days'][ii])))
print('')
for ii in range(len(country)):
    print('The expected deaths in '+country[ii]+' in '+ str(t_analysis) + ' days are ' +  str(int(data['deaths_in_t_days'][ii])))
