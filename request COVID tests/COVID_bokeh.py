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
data_case=data_case.fillna(value=0)
#data_case=data_case.to_dict('series')

#data_case[np.isnan(data_case)] = 0
s = requests.get(url_d).content
data_death = pd.read_csv(io.StringIO(s.decode('utf-8')))
data_death.fillna(value=0)
#data_death[np.isnan(data_death)] = 0

#t_analysis = 2 # days
#country = ['Germany','Portugal', 'France']
country = ['Italy','Spain','France','Germany','United Kingdom', 'Switzerland']

#date_c = data_case['date']
#date_d = data_death['date']



#data = analysis_country(data_death, data_case,country, t_analysis)



#%% Bokeh interactive image
from bokeh.plotting import figure, output_file, show, ColumnDataSource, save
from bokeh.models.tools import HoverTool
from bokeh.io import export_png
from bokeh.embed import components
from bokeh.resources import CDN
from bokeh.embed import file_html

def dict_prep(data_case, country, data_death):
    dates_c = data_case['date']
    data_case = data_case[country]
    sum_cases = np.sum(data_case)
    filename_c = []
    
    dates_d = data_death['date']
    data_death = data_death[country]
    sum_death = np.sum(data_death)
    filename_d_div = []
    filename_d_script = []
    for ii in range(len(country)):
        fig1 = figure(y_range=(0, np.max(data_case[country[ii]])),
                      x_range=(0, len(dates_c)),
                      plot_width=400,
                      plot_height=200,
                      x_axis_label = '# days',
                      title = 'New cases per day ' + country[ii],
                      tools="")

        fig1.vbar(x=np.arange(0,len(dates_c)),top=data_case[country[ii]],width=.9,color='orange')
        output_file(country[ii]+'_c.html')
        save(fig1)
        filename_c += [country[ii]+'_c.png']
#        export_png(fig1, filename_c[ii])
        
        fig2 = figure(y_range=(0, np.max(data_death[country[ii]])),
                      x_range=(0, len(dates_d)),
                      plot_width=400,
                      plot_height=200,
                      x_axis_label = '# days',
                      title = 'New deaths per day ' + country[ii],
                      tools="")
        fig2.vbar(x=np.arange(0,len(dates_d)),top=data_death[country[ii]],width=.9,color='blue',)
        output_file(country[ii]+'_d.html')
        save(fig2)
        script, div = components(fig2)
#        filename_d += [country[ii]+'_d.png']
#        export_png(fig2, filename_d[ii])
        filename_d_script += [script]
        filename_d_div += [div]

    return {'sum_cases':sum_cases, 'country':country, 'cases':data_case, 'filename_c':filename_c,
            'sum_death':sum_death, 'death':data_death, 'filename_d_script':filename_d_script, 'filename_d_div':filename_d_div,
            'percentage_death':np.array((sum_death/sum_cases*100),dtype='int')}



data = dict_prep(data_case, country,data_death)
source = ColumnDataSource(data)

##country_range = source.data['country'].tolist()


fig = figure(y_range=country,
             plot_width=800,
             plot_height=600,
             x_axis_label = 'Total number of cases (deaths)',
             title = 'Covid analysis')



#y = data['country']
#right=data['sum_cases']
#right = np.array(np.sum(data_case[country]))

fig.hbar(y='country',
         right='sum_cases',
         height=.4,
         left=0,
         color="Orange",
         fill_alpha=.5,
         source=source)



fig.hbar(y='country',
         right='sum_death',
         height=.2,
         left=0,
         color="Blue",
         fill_alpha=.5,
         source=source)


#Hover tools
hover = HoverTool()
hover.tooltips= """
<html>
<head>
<meta charset="utf-8">
        <title>Bokeh Scatter Plots</title>

        <script src="https://cdn.bokeh.org/bokeh/release/bokeh-1.1.0.min.js"></script>
<script type="text/javascript">\n  (function() {\n    var fn = function() {\n      Bokeh.safely(function() {\n        (function(root) {\n          function embed_document(root) {\n            \n          var docs_json = \'{"edfb049e-3dd3-4ad2-98b0-8ae80c9ed641":{"roots":{"references":[{"attributes":{},"id":"124803","type":"BasicTicker"},{"attributes":{"callback":null,"end":971},"id":"124796","type":"Range1d"},{"attributes":{"text":"New deaths per day Italy"},"id":"124792","type":"Title"},{"attributes":{"formatter":{"id":"124822","type":"BasicTickFormatter"},"ticker":{"id":"124808","type":"BasicTicker"}},"id":"124807","type":"LinearAxis"},{"attributes":{"source":{"id":"124813","type":"ColumnDataSource"}},"id":"124817","type":"CDSView"},{"attributes":{"callback":null,"end":134},"id":"124794","type":"Range1d"},{"attributes":{"active_drag":"auto","active_inspect":"auto","active_multi":null,"active_scroll":"auto","active_tap":"auto"},"id":"124812","type":"Toolbar"},{"attributes":{"below":[{"id":"124802","type":"LinearAxis"}],"center":[{"id":"124806","type":"Grid"},{"id":"124811","type":"Grid"}],"left":[{"id":"124807","type":"LinearAxis"}],"plot_height":200,"plot_width":400,"renderers":[{"id":"124816","type":"GlyphRenderer"}],"title":{"id":"124792","type":"Title"},"toolbar":{"id":"124812","type":"Toolbar"},"x_range":{"id":"124794","type":"Range1d"},"x_scale":{"id":"124798","type":"LinearScale"},"y_range":{"id":"124796","type":"Range1d"},"y_scale":{"id":"124800","type":"LinearScale"}},"id":"124791","subtype":"Figure","type":"Plot"},{"attributes":{},"id":"124820","type":"BasicTickFormatter"},{"attributes":{},"id":"124798","type":"LinearScale"},{"attributes":{"data_source":{"id":"124813","type":"ColumnDataSource"},"glyph":{"id":"124814","type":"VBar"},"hover_glyph":null,"muted_glyph":null,"nonselection_glyph":{"id":"124815","type":"VBar"},"selection_glyph":null,"view":{"id":"124817","type":"CDSView"}},"id":"124816","type":"GlyphRenderer"},{"attributes":{},"id":"124822","type":"BasicTickFormatter"},{"attributes":{"dimension":1,"ticker":{"id":"124808","type":"BasicTicker"}},"id":"124811","type":"Grid"},{"attributes":{},"id":"124823","type":"Selection"},{"attributes":{"callback":null,"data":{"top":[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,4,5,1,5,4,8,6,17,28,27,41,49,36,133,98,167,196,189,252,173,370,347,347,473,429,625,795,649,601,743,685,660,971,887,758,810,839,727,760,764,681,527,636,604,540,612,570,619,431,564,604,578,525,575,480,433,454,534,437,464,420,415,260,333,382,323,285,269,474,174,195,236,369,274,243,194,165,179],"x":[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,132,133]},"selected":{"id":"124823","type":"Selection"},"selection_policy":{"id":"124824","type":"UnionRenderers"}},"id":"124813","type":"ColumnDataSource"},{"attributes":{},"id":"124800","type":"LinearScale"},{"attributes":{"axis_label":"# days","formatter":{"id":"124820","type":"BasicTickFormatter"},"ticker":{"id":"124803","type":"BasicTicker"}},"id":"124802","type":"LinearAxis"},{"attributes":{"ticker":{"id":"124803","type":"BasicTicker"}},"id":"124806","type":"Grid"},{"attributes":{"fill_color":{"value":"blue"},"line_color":{"value":"blue"},"top":{"field":"top"},"width":{"value":0.9},"x":{"field":"x"}},"id":"124814","type":"VBar"},{"attributes":{},"id":"124824","type":"UnionRenderers"},{"attributes":{"fill_alpha":{"value":0.1},"fill_color":{"value":"#1f77b4"},"line_alpha":{"value":0.1},"line_color":{"value":"#1f77b4"},"top":{"field":"top"},"width":{"value":0.9},"x":{"field":"x"}},"id":"124815","type":"VBar"},{"attributes":{},"id":"124808","type":"BasicTicker"}],"root_ids":["124791"]},"title":"Bokeh Application","version":"1.2.0"}}\';\n          var render_items = [{"docid":"edfb049e-3dd3-4ad2-98b0-8ae80c9ed641","roots":{"124791":"e6ac5739-b689-4b10-a9de-fb0a8b994967"}}];\n          root.Bokeh.embed.embed_items(docs_json, render_items);\n        \n          }\n          if (root.Bokeh !== undefined) {\n            embed_document(root);\n          } else {\n            var attempts = 0;\n            var timer = setInterval(function(root) {\n              if (root.Bokeh !== undefined) {\n                embed_document(root);\n                clearInterval(timer);\n              }\n              attempts++;\n              if (attempts > 100) {\n                console.log("Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing");\n                clearInterval(timer);\n              }\n            }, 10, root)\n          }\n        })(window);\n      });\n    };\n    if (document.readyState != "loading") fn();\n    else document.addEventListener("DOMContentLoaded", fn);\n  })();\n</script>

<style>
div.fixed {
  position: fixed;
  bottom: 20px;
  right: 20px;
  width: 410px;
  border: 3px solid black;
  background: white;
}
</style>
</head>
<body>
<div class="bk-root" id="7ff076c8-e72b-4b99-ad05-7a138ce94a70" data-root-id="126202"></div>

<div class="fixed">
    <h3>@country</h3>
    <div><strong>Country: </strong>@country</div>
    <div><strong>Total number of cases: </strong>@sum_cases</div>
    <div><strong>Total number of deaths: </strong>@sum_death</div>
    <div><strong>Death percentage: </strong>@percentage_death %</div>
    <div><img src="@filename_c" alt="" width="400" /></div>
</div>
</body>
</html>

"""

fig.add_tools(hover)
#fig.add_glyph(source, data['filename_d'][2])

output_file('bar.html')

show(fig)
script, div = components(fig)
