#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 15 18:21:20 2020

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
    I0, R0 = 9, 0
    # Everyone else, S0, is susceptible to infection initially.
    S0 = N - I0 - R0
    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
    # Initial conditions vector
    y0 = S0, I0, R0
    # Integrate the SIR equations over the time grid, t.
    ret = odeint(deriv, y0, t, args=(N, beta, gamma))
    S, I, R = ret.T
    return I

def new_cases(total_cases):
    if len(total_cases)==0:
        new_cases_array = 0
    else:
        new_cases_array = np.zeros(len(total_cases))
        new_cases_array[0] = 0
        for ii in range(len(total_cases)-1):
            new_cases_array[ii+1] = total_cases[ii+1]-total_cases[ii]
    return new_cases_array

def analysis_country(CCAA,data,fit):
    fallecidos = data['Fallecidos']
    casos = data['CASOS']
    
    fallecidos[np.isnan(fallecidos)] = 0
    casos[np.isnan(casos)] = 0
    dates = data[keys_data_dict[1]]
    cat = np.where(data['CCAA']==CCAA)[0]

    fallecidos_cat = np.array(fallecidos[cat])
    dates_cat = np.array(dates[cat])
    casos_cat = np.array(casos[cat])
    
    nuevos_fallecidos_cat = new_cases(fallecidos_cat)
    nuevos_casos_cat = new_cases(casos_cat)
    
    
    if fit:
        # Defining Splits
        splits = 3
        # Finding average of each consecutive segment
        nuevos_fallecidos_cat_splits = [sum(nuevos_fallecidos_cat[i:i + splits])/splits for i in range(len(nuevos_fallecidos_cat) - splits + 1)]
        nuevos_fallecidos_cat_splits = np.array(nuevos_fallecidos_cat_splits,dtype='int')
        threashold = 10#np.max(cases)/100
#        t0_nuevos_fallecidos_cat = np.where(nuevos_fallecidos_cat_splits>threashold)[0][0]
#        nuevos_fallecidos_cat_2fit=np.array(nuevos_fallecidos_cat_splits)[t0_nuevos_fallecidos_cat:-1]
#        time_fit_d = np.linspace(t0_nuevos_fallecidos_cat,len(nuevos_fallecidos_cat_splits),len(nuevos_fallecidos_cat_2fit))
#        p0 = [np.max(nuevos_fallecidos_cat)*2,0.4,1/10]
#        popt_d, pcov_d = curve_fit(sir_model, time_fit_d, nuevos_fallecidos_cat_2fit, p0, bounds=([0,0,0],[np.inf,np.inf,np.inf]))
        t0_nuevos_fallecidos_cat = np.where(nuevos_fallecidos_cat>threashold)[0][0]
        nuevos_fallecidos_cat_2fit=np.array(nuevos_fallecidos_cat)[t0_nuevos_fallecidos_cat:len(nuevos_fallecidos_cat)]
        time_fit_d = np.linspace(t0_nuevos_fallecidos_cat,len(nuevos_fallecidos_cat)-1,len(nuevos_fallecidos_cat_2fit),dtype='int')
        p0 = [np.max(nuevos_fallecidos_cat)*2,0.4,1/10]
        popt_d, pcov_d = curve_fit(sir_model, time_fit_d, nuevos_fallecidos_cat_2fit, p0, bounds=([0,0,0],[np.inf,np.inf,np.inf]))

        # Defining Splits
        splits = 3
        # Finding average of each consecutive segment
        nuevos_casos_cat_splits = [sum(nuevos_casos_cat[i:i + splits])/splits for i in range(len(nuevos_casos_cat) - splits + 1)]
        nuevos_casos_cat_splits = np.array(nuevos_casos_cat_splits,dtype='int')
        threashold = 10#np.max(cases)/100
        t0_nuevos_casos_cat = np.where(nuevos_casos_cat>threashold)[0][0]
        nuevos_casos_cat_2fit=np.array(nuevos_casos_cat)[t0_nuevos_casos_cat:len(nuevos_casos_cat)]
        time_fit_c = np.linspace(t0_nuevos_casos_cat,len(nuevos_casos_cat)-1,len(nuevos_casos_cat_2fit),dtype='int')
        p0 = [np.max(nuevos_casos_cat)*20,0.22,1/12]
        popt_c, pcov_c = curve_fit(sir_model, time_fit_c, nuevos_casos_cat_2fit, p0, bounds=([0,0,0],[np.inf,np.inf,np.inf]))

        
    
    return{'dates': dates_cat, 'death_tot': fallecidos_cat, 'death_new': nuevos_fallecidos_cat, 'death_new_smooth':nuevos_fallecidos_cat_splits,
           'death_fit_params':popt_d, 'death_fit_errors':pcov_d, 'death_fit_from_day':time_fit_d,
           'cases_tot': casos_cat, 'cases_new': nuevos_casos_cat, 'cases_new_smooth':nuevos_casos_cat_splits,
           'cases_fit_params':popt_c, 'cases_fit_errors':pcov_c, 'cases_fit_from_day':time_fit_c}


def analysis_what(CCAA,data,what,fit,interval_restrictions):
    fallecidos = data[what]
    fallecidos[np.isnan(fallecidos)] = 0

    dates = data[keys_data_dict[1]]
    cat = np.where(data['CCAA']==CCAA)[0]

    fallecidos_cat = np.array(fallecidos[cat])
    dates_cat = np.array(dates[cat])
    
    nuevos_fallecidos_cat = new_cases(fallecidos_cat)
    
    
    if fit:
        # Defining Splits
        splits = 3
        # Finding average of each consecutive segment
        nuevos_fallecidos_cat_splits = [sum(nuevos_fallecidos_cat[i:i + splits])/splits for i in range(len(nuevos_fallecidos_cat) - splits + 1)]
        nuevos_fallecidos_cat_splits = np.array(nuevos_fallecidos_cat_splits,dtype='int')
        threashold = 10#np.max(cases)/100
#        t0_nuevos_fallecidos_cat = np.where(nuevos_fallecidos_cat_splits>threashold)[0][0]
#        nuevos_fallecidos_cat_2fit=np.array(nuevos_fallecidos_cat_splits)[t0_nuevos_fallecidos_cat:-1]
#        time_fit_d = np.linspace(t0_nuevos_fallecidos_cat,len(nuevos_fallecidos_cat_splits),len(nuevos_fallecidos_cat_2fit))
#        p0 = [np.max(nuevos_fallecidos_cat)*2,0.4,1/10]
#        popt_d, pcov_d = curve_fit(sir_model, time_fit_d, nuevos_fallecidos_cat_2fit, p0, bounds=([0,0,0],[np.inf,np.inf,np.inf]))
        t0_nuevos_fallecidos_cat = np.where(nuevos_fallecidos_cat>threashold)[0][0]
        nuevos_fallecidos_cat_2fit=np.array(nuevos_fallecidos_cat)[t0_nuevos_fallecidos_cat:len(nuevos_fallecidos_cat)]
        time_fit_d = np.linspace(t0_nuevos_fallecidos_cat,len(nuevos_fallecidos_cat)-1,len(nuevos_fallecidos_cat_2fit),dtype='int')
        p0 = [np.max(nuevos_fallecidos_cat)*2,0.4,1/10]
        popt_d, pcov_d = curve_fit(sir_model, time_fit_d, nuevos_fallecidos_cat_2fit, p0, bounds=([0,0,0],[np.inf,np.inf,np.inf]))


    return{'dates': dates_cat, 'death_tot': fallecidos_cat, 'death_new': nuevos_fallecidos_cat, 'death_new_smooth':nuevos_fallecidos_cat_splits,
           'death_fit_params':popt_d, 'death_fit_errors':pcov_d, 'death_fit_from_day':time_fit_d}


url_spain = 'https://covid19.isciii.es/resources/serie_historica_acumulados.csv'
s = requests.get(url_spain).content
s=s[0:-1357-373]
data = pd.read_csv(io.StringIO(s.decode('utf-8')))
keys_data_dict=['CCAA', 'FECHA', 'CASOS', 'Hospitalizados', 'UCI', 'Fallecidos',
       'Recuperados']
dates = data[keys_data_dict[1]]

##%%
#fallecidos = data['Fallecidos']
#fallecidos[np.isnan(fallecidos)] = 0
#cat = np.where(data['CCAA']=='CT')[0]
#
#fallecidos_cat = np.array(fallecidos[cat])
#plt.plot(fallecidos[cat])
#plt.plot(data['CASOS'][cat])
#
##%%
#nuevos_fallecidos_cat = new_cases(fallecidos_cat)
#
## Defining Splits
#splits = 1
## Finding average of each consecutive segment
#nuevos_fallecidos_cat = [sum(nuevos_fallecidos_cat[i:i + splits])/splits for i in range(len(nuevos_fallecidos_cat) - splits + 1)]
#nuevos_fallecidos_cat = np.array(nuevos_fallecidos_cat,dtype='int')
#threashold = 0#np.max(cases)/100
#nuevos_fallecidos_cat_2fit=np.array(nuevos_fallecidos_cat)[np.where(nuevos_fallecidos_cat>threashold)[0]]
#time_fit_c = np.where(nuevos_fallecidos_cat>threashold)[0]
#t0_nuevos_fallecidos_cat=time_fit_c[0]
#p0 = [np.max(nuevos_fallecidos_cat)*20,0.22,1/12]
#popt_c, pcov_c = curve_fit(sir_model, time_fit_c, nuevos_fallecidos_cat_2fit, p0)
#
#x_axis = np.linspace(time_fit_c[0],time_fit_c[-1]+10,len(time_fit_c)+10)
#plt.plot(time_fit_c,nuevos_fallecidos_cat_2fit,'o')
#plt.plot(x_axis,sir_model(x_axis,*popt_c))


#%%
dates_change_restrictions = []
dates_change_restrictions.append('16/3/2020')
dates_change_restrictions.append('14/4/2020')

CATdat = analysis_what('CT',data,'CASOS',True,dates_change_restrictions)
dates_cat = CATdat['dates']

indx_dates_change_restrictions = np.zeros(len(dates_change_restrictions))
indx_dates_change_restrictions[0] = np.where(dates_cat==dates_change_restrictions[0])[0]# = indx_dates_change_restrictions+np.where(dates_cat=='16/3/2020')[0]
indx_dates_change_restrictions[1] = np.where(dates_cat==dates_change_restrictions[1])[0]

#c_tfit = CATdat['cases_fit_from_day']
#c_axis = np.linspace(c_tfit[0],c_tfit[-1]+10,len(c_tfit)+10)-c_tfit[0]


#indx_dates_closure_restrictions = np.where(dates_cat=='16/3/2020')[0]
#indx_dates_less_restrictions = np.where(dates_cat=='14/4/2020')[0]

#plt.figure(0)
#plt.clf()
#plt.plot(CATdat['cases_new'][c_tfit],'o')
#plt.plot(CATdat['cases_new_smooth'][c_tfit],'.')
#plt.plot(c_axis,sir_model(c_axis,*CATdat['cases_fit_params']))
#plt.plot(np.linspace(inds_dates_less_restrictions-c_tfit[0],inds_dates_less_restrictions-c_tfit[0],2),
#np.linspace(0,np.max(CATdat['cases_new']),2),'--')
#plt.xlabel('days from '+str(dates_cat[c_tfit[0]]))
#plt.ylabel('New cases per day [CAT]')
#plt.legend(['raw data','smooth data','fit','less restrictions from here'])

d_tfit = CATdat['death_fit_from_day']
d_axis = np.linspace(d_tfit[0],d_tfit[-1]+10,len(d_tfit)+10)-d_tfit[0]

#plt.figure(1)
#plt.clf()
#plt.plot(CATdat['death_new'][d_tfit[0]:-1],'o')
#plt.plot(CATdat['death_new_smooth'][d_tfit[0]:-1],'.')
#plt.plot(d_axis,sir_model(d_axis,*CATdat['death_fit_params']))
#for ii in range(len(indx_dates_change_restrictions)):
#    plt.plot(np.linspace(indx_dates_change_restrictions[ii]-d_tfit[0],indx_dates_change_restrictions[ii]-d_tfit[0],2),
#np.linspace(0,np.max(CATdat['death_new']),2),'--')
#plt.xlabel('days from '+str(dates_cat[d_tfit[0]]))
#plt.ylabel('New deaths per day [CAT]')
#plt.legend(['raw data','smooth data','fit','less restrictions from here'])

d_axis = np.linspace(d_tfit[0],d_tfit[-1]+10,len(d_tfit)+10)-d_tfit[0]
plt.figure(1)
plt.clf()
plt.plot(np.arange(len(CATdat['death_new']))-d_tfit[0], CATdat['death_new'],'o')
plt.plot(np.arange(len(CATdat['death_new_smooth']))-d_tfit[0]+(len(CATdat['death_new'])-len(CATdat['death_new_smooth']))/2,CATdat['death_new_smooth'],'.')
plt.plot(d_axis,sir_model(d_axis,*CATdat['death_fit_params']))
for ii in range(len(indx_dates_change_restrictions)):
    plt.plot(np.linspace(indx_dates_change_restrictions[ii]-d_tfit[0],indx_dates_change_restrictions[ii]-d_tfit[0],2),
np.linspace(0,np.max(CATdat['death_new']),2),'--')
plt.xlabel('days from '+str(dates_cat[d_tfit[0]]))
plt.ylabel('New deaths per day [CAT]')
plt.legend(['raw data','smooth data','fit','less restrictions from here'])
plt.xlim([0,d_axis[-1]])

#c_axis = np.linspace(c_tfit[0],c_tfit[-1]+10,len(c_tfit)+10)-c_tfit[0]
#plt.figure(0)
#plt.clf()
#plt.plot(np.arange(len(CATdat['cases_new']))-c_tfit[0], CATdat['cases_new'],'o')
#plt.plot(np.arange(len(CATdat['cases_new_smooth']))-c_tfit[0]+(len(CATdat['cases_new'])-len(CATdat['cases_new_smooth']))/2,CATdat['cases_new_smooth'],'.')
#plt.plot(c_axis,sir_model(c_axis,*CATdat['cases_fit_params']))
#for ii in range(len(indx_dates_change_restrictions)):
#    plt.plot(np.linspace(indx_dates_change_restrictions[ii]-c_tfit[0],indx_dates_change_restrictions[ii]-c_tfit[0],2),
#np.linspace(0,np.max(CATdat['cases_new']),2),'--')
#plt.xlabel('days from '+str(dates_cat[c_tfit[0]]))
#plt.ylabel('New deaths per day [CAT]')
#plt.legend(['raw data','smooth data','fit','less restrictions from here'])
#plt.xlim([0,c_axis[-1]])

#%%


##%%
#
#CCAA = 'CT'
#fit = True
#fallecidos = data['Fallecidos']
#casos = data['CASOS']
#
#fallecidos[np.isnan(fallecidos)] = 0
#casos[np.isnan(casos)] = 0
#dates = data[keys_data_dict[1]]
#cat = np.where(data['CCAA']==CCAA)[0]
#
#fallecidos_cat = np.array(fallecidos[cat])
#dates_cat = np.array(dates[cat])
#casos_cat = np.array(casos[cat])
#
#nuevos_fallecidos_cat = new_cases(fallecidos_cat)
#nuevos_casos_cat = new_cases(casos_cat)
#
#
#if fit:
#    # Defining Splits
#    splits = 3
#    # Finding average of each consecutive segment
#    nuevos_fallecidos_cat_splits = [sum(nuevos_fallecidos_cat[i:i + splits])/splits for i in range(len(nuevos_fallecidos_cat) - splits + 1)]
#    nuevos_fallecidos_cat_splits = np.array(nuevos_fallecidos_cat_splits,dtype='int')
#    threashold = 0#np.max(cases)/100
##    t0_nuevos_fallecidos_cat = np.where(nuevos_fallecidos_cat_splits>threashold)[0][0]
##    nuevos_fallecidos_cat_2fit=np.array(nuevos_fallecidos_cat_splits)[t0_nuevos_fallecidos_cat:len(nuevos_fallecidos_cat_splits)]
##    time_fit_d = np.linspace(t0_nuevos_fallecidos_cat+1,len(nuevos_fallecidos_cat_splits),len(nuevos_fallecidos_cat_splits)-t0_nuevos_fallecidos_cat)
##    p0 = [np.max(nuevos_fallecidos_cat)*2,0.4,1/10]
##    popt_d, pcov_d = curve_fit(sir_model, time_fit_d, nuevos_fallecidos_cat_2fit, p0, bounds=([0,0,0],[np.inf,np.inf,np.inf]))
#    t0_nuevos_fallecidos_cat = np.where(nuevos_fallecidos_cat>threashold)[0][0]
#    nuevos_fallecidos_cat_2fit=np.array(nuevos_fallecidos_cat)[t0_nuevos_fallecidos_cat:-1]
#    time_fit_d = np.linspace(t0_nuevos_fallecidos_cat,len(nuevos_fallecidos_cat),len(nuevos_fallecidos_cat_2fit))
#    p0 = [np.max(nuevos_fallecidos_cat)*2,0.4,1/10]
#    popt_d, pcov_d = curve_fit(sir_model, time_fit_d, nuevos_fallecidos_cat_2fit, p0, bounds=([0,0,0],[np.inf,np.inf,np.inf]))

#%%
#
## here there is a constant increase of the number of possible infected people 
#def deriv_mu(y, t, N, beta, gamma, mu):
#    S, I, R = y
#    dSdt = -beta * S * I / N + mu
#    dIdt = beta * S * I / N - gamma * I
#    dRdt = gamma * I
#    return dSdt, dIdt, dRdt
#
#
#def sir_model_mu(t, N, beta, gamma,mu):
#    # Total population, N.
#    # Initial number of infected and recovered individuals, I0 and R0.
#    I0, R0 = 1, 0
#    # Everyone else, S0, is susceptible to infection initially.
#    S0 = N - I0 - R0
#    # Contact rate, beta, and mean recovery rate, gamma, (in 1/days).
#    # Initial conditions vector
#    y0 = S0, I0, R0
#    # Integrate the SIR equations over the time grid, t.
#    ret = odeint(deriv_mu, y0, t, args=(N, beta, gamma,mu))
#    S, I, R = ret.T
#    return I
#
#
#def analysis_country_mu(CCAA,data,fit='True'):
#    fallecidos = data['Fallecidos']
#    casos = data['CASOS']
#    
#    fallecidos[np.isnan(fallecidos)] = 0
#    casos[np.isnan(casos)] = 0
#    dates = data[keys_data_dict[1]]
#    cat = np.where(data['CCAA']==CCAA)[0]
#
#    fallecidos_cat = np.array(fallecidos[cat])
#    dates_cat = np.array(dates[cat])
#    casos_cat = np.array(casos[cat])
#    
#    nuevos_fallecidos_cat = new_cases(fallecidos_cat)
#    nuevos_casos_cat = new_cases(casos_cat)
#    
#    
#    if fit:
#        # Defining Splits
#        splits = 3
#        # Finding average of each consecutive segment
#        nuevos_fallecidos_cat_splits = [sum(nuevos_fallecidos_cat[i:i + splits])/splits for i in range(len(nuevos_fallecidos_cat) - splits + 1)]
#        nuevos_fallecidos_cat_splits = np.array(nuevos_fallecidos_cat_splits,dtype='int')
#        threashold = 0#np.max(cases)/100
#        t0_nuevos_fallecidos_cat = np.where(nuevos_fallecidos_cat>threashold)[0][0]
#        nuevos_fallecidos_cat_2fit=np.array(nuevos_fallecidos_cat)[t0_nuevos_fallecidos_cat:len(nuevos_fallecidos_cat)]
#        time_fit_d = np.linspace(t0_nuevos_fallecidos_cat,len(nuevos_fallecidos_cat)-1,len(nuevos_fallecidos_cat_2fit),dtype='int')
#        p0 = [np.max(nuevos_fallecidos_cat)*2,0.4,1/10,0]
#        popt_d, pcov_d = curve_fit(sir_model_mu, time_fit_d, nuevos_fallecidos_cat_2fit, p0, bounds=([0,0,0,0],[np.inf,np.inf,np.inf,np.inf]))
#
#        # Defining Splits
#        splits = 3
#        # Finding average of each consecutive segment
#        nuevos_casos_cat_splits = [sum(nuevos_casos_cat[i:i + splits])/splits for i in range(len(nuevos_casos_cat) - splits + 1)]
#        nuevos_casos_cat_splits = np.array(nuevos_casos_cat_splits,dtype='int')
#        threashold = 0#np.max(cases)/100
#        t0_nuevos_casos_cat = np.where(nuevos_casos_cat>threashold)[0][0]
#        nuevos_casos_cat_2fit=np.array(nuevos_casos_cat)[t0_nuevos_casos_cat:len(nuevos_casos_cat)]
#        time_fit_c = np.linspace(t0_nuevos_casos_cat,len(nuevos_casos_cat)-1,len(nuevos_casos_cat_2fit),dtype='int')
#        p0 = [np.max(nuevos_casos_cat)*20,0.22,1/12,0]
#        popt_c, pcov_c = curve_fit(sir_model_mu, time_fit_c, nuevos_casos_cat_2fit, p0, bounds=([0,0,0,0],[np.inf,np.inf,np.inf,np.inf]))
#
#        
#    
#    return{'dates': dates_cat, 'death_tot': fallecidos_cat, 'death_new': nuevos_fallecidos_cat, 'death_new_smooth':nuevos_fallecidos_cat_splits,
#           'death_fit_params':popt_d, 'death_fit_errors':pcov_d, 'death_fit_from_day':time_fit_d,
#           'cases_tot': casos_cat, 'cases_new': nuevos_casos_cat, 'cases_new_smooth':nuevos_casos_cat_splits,
#           'cases_fit_params':popt_c, 'cases_fit_errors':pcov_c, 'cases_fit_from_day':time_fit_c}
