import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
# Load the Pandas libraries with alias 'pd' 
import pandas as pd 
import io

import requests
import json


s='records_as_of_2021_01_10_CET.csv'
columns_to_use = ['Study Country', 'Site Status']
data_csv = pd.read_csv(s, usecols=columns_to_use)
data_csv.head()
data_csv_keys = data_csv.columns

#rename columns
data_csv.columns = ['Country','Site_Status']
import pycountry
abbr_vec = []
for ii in range(len(data_csv)):
    abbr = pycountry.countries.search_fuzzy(data_csv['Country'][ii])
#    print(abbr)
    abbr_vec= abbr_vec + [abbr[0].alpha_3]
data_csv['Country Abbr']=abbr_vec
data_csv.head()
data_csv.drop('Country', axis=1)

import geopandas as gpd
from geonamescache import GeonamesCache

shapefile = 'map/ne_10m_admin_0_countries_lakes.shp'
gdf = gpd.read_file(shapefile)[['ADMIN', 'ADM0_A3', 'geometry']]
#Rename columns.
gdf.columns = ['country', 'country_code', 'geometry']

##Drop row corresponding to 'Antarctica'
print(gdf[gdf['country'] == 'Antarctica'])
gdf = gdf.drop(gdf.index[159])

gdf.set_index('country_code', inplace=True)
gdf.head()



di = {'Did Not Participate': 0, 'Active': 1, 'Terminated': 2}
data_csv['status'] = data_csv['Site_Status']
data_csv['Site_Status'] = data_csv['Site_Status'].map(di)

#my dataframe
data_csv.set_index('Country Abbr', inplace=True)
#data_csv = data_csv.ix[gdf['country_code']].dropna() # Filter out non-countries and missing values.

data_csv.head()


#%% create json file megring country codes and value to plot
#Merge dataframes.
merged = pd.concat([gdf, data_csv], axis=1, sort=True)
#merged['Site_Status'] = merged['Site_Status'].fillna(0)
merged.fillna('No data', inplace = True)

#Read data to json.
merged_json = json.loads(merged.to_json())

#Convert to String like object.
json_data = json.dumps(merged_json)

#%% Bokeh integration for interactive map
from bokeh.io import output_notebook, show, output_file
from bokeh.plotting import figure
from bokeh.models import GeoJSONDataSource, LinearColorMapper, ColorBar
from bokeh.palettes import brewer#Input GeoJSON source that contains features for plotting.

# Let's put some interactivity
from bokeh.io import curdoc
from bokeh.models import Slider, HoverTool
from bokeh.layouts import widgetbox, row, column

geosource = GeoJSONDataSource(geojson = json_data)
#Define a sequential multi-hue color palette.
palette = brewer['YlGnBu'][8]

#Instantiate LinearColorMapper that linearly maps numbers in a range, into a sequence of colors.
color_mapper = LinearColorMapper(palette = palette, low = 0, high = 2, nan_color = '#d9d9d9')
#Define custom tick labels for color bar.
tick_labels = {'0': 'Did not participate', '1':'Active', '2':'Terminated'}#Create color bar. 
color_bar = ColorBar(color_mapper=color_mapper, label_standoff=8,width = 500, height = 20,
border_line_color=None,location = (0,0), orientation = 'horizontal', major_label_overrides = tick_labels)

#Add hover tool
hover = HoverTool(tooltips = [ ('Country/region','@country'),('Status trial', '@status')])

#Create figure object.
p = figure(title = 'Status of sites for phase III clinical trial', plot_height = 600 , plot_width = 950,
           toolbar_location = None, tools = [hover])
p.xgrid.grid_line_color = None
p.ygrid.grid_line_color = None
#Add patch renderer to figure. 
p.patches('xs','ys', source = geosource,fill_color = {'field' :'Site_Status', 'transform' : color_mapper},
          line_color = 'black', line_width = 0.25, fill_alpha = 1)
#Specify figure layout.
p.add_layout(color_bar, 'below')
#Display figure inline in Jupyter Notebook.
output_notebook()
#Display figure.
show(p)
output_file("foo.html")