# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
sns.set_style('whitegrid')
import plotly.express as px
import os 
import glob
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
from streamlit_lottie import st_lottie

# Importing the Data 

data = pd.read_csv(r"F:\Data Science Projects\ESG Book\artifacts\ESG_cleaned_data.csv")
data.set_index(['isin','country','year'],inplace=True,drop=True)

emission_metrics = pd.read_csv(r"F:\Data Science Projects\ESG Book\data\Emissions_metrics_data_dictionary.csv")
emission_metrics = emission_metrics.iloc[:19,:]
emission_metrics.set_index('Key',inplace=True,drop=True)

em_metric_names = data.columns.to_list()
company_names = data.index.levels[0].to_list()
country_names = data.index.levels[1].to_list() + ['All Countries']
years = data.index.levels[2].to_list()

# Exploratory Data Analytics

st.set_page_config(layout='wide')
st.title('ESG Insights Explorer')
analytics_gif = "Analytics_GIF.gif"
st.image(analytics_gif)
st.subheader('An App by Aniruddha Prabhu')
st.write('Introducing the ESG Insights Explorer, your go-to app for streamlined exploration of Environmental, Social, and Governance (ESG) data. This user-friendly tool aggregates comprehensive ESG metrics from diverse sources, offering a customizable dashboard for personalized insights. Stay up-to-date with real-time updates, benchmark performance against industry standards, and conduct trend analysis for informed decision-making. The app facilitates risk assessment, measures positive impact, and encourages collaboration through shared insights. With educational resources included, the ESG Insights Explorer empowers investors, businesses, and individuals to make impactful, sustainable choices in an ever-evolving landscape.')

### Plot-1
st.subheader("Number of Companies sorted by Country")

country_count = dict()
cc = (data.groupby(['country'])['em_60000'].count()/11).astype('int').sort_values(ascending=False).to_dict()
country_count['Country'] = list(cc.keys())
country_count['No_Comp'] = list(cc.values())
df_country_count = pd.DataFrame(country_count)

fig1 = px.treemap(df_country_count, 
                 path=['Country'], 
                 values='No_Comp', 
                 title='Number of Companies Sorted by Country')

st.plotly_chart(fig1)

### Plot-2
st.subheader('Mean Emissions across Countries or Companies: 2013 to 2023')

sort_plot2 = st.selectbox('Select an option to sort by',['Country','Company'],key='sort_plot2')
em_metric_sel_plot2 = st.selectbox('Select the Emission Metric you want to assess',em_metric_names,key='em_metric_sel')

if sort_plot2 == 'Country':
    sort_plot2_sel = 'country'
else:
    sort_plot2_sel = 'isin'

em_metrics = dict()
em = data.groupby([sort_plot2_sel])[em_metric_sel_plot2].mean().sort_values(ascending=False).to_dict()
em_metrics[sort_plot2_sel] = list(em.keys())
em_metrics['EM'] = list(em.values())
df_em_metrics = pd.DataFrame(em_metrics)

fig2 = px.treemap(df_em_metrics, 
                 path=[sort_plot2_sel], 
                 values='EM', 
                 title=f'Mean Emissions for {em_metric_sel_plot2} Across {sort_plot2}: 2013 to 2023')

st.plotly_chart(fig2)

### Plot-3
st.subheader('Mean Distribution of Emission Metrics across Countries or Companies: 2013 to 2023')

sort_plot3 = st.selectbox('Select an option to sort by',['Country','Company'],key='sort_plot3')

if sort_plot3 == 'Country':
    sort_plot3_sel = 'country'
    subcat_plot3 = data.index.levels[1].to_list()
else:
    sort_plot3_sel = 'isin'
    subcat_plot3 = data.index.levels[0].to_list()   

subcat_plot3_sel = st.selectbox('Select the sub-category',subcat_plot3,key='subcat_plot3_sel')

em_total = data.groupby(sort_plot3_sel)[em_metric_names].mean().loc[subcat_plot3_sel,:].sum()
em_ind = data.groupby(sort_plot3_sel)[em_metric_names].mean().loc[subcat_plot3_sel,:]
em_percent = np.round((em_ind*100)/em_total,2)

categories = em_percent.index.to_list()
colors = plt.get_cmap('tab20c').colors
category_colors = colors[:len(categories)]
explode = [ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0]
fontsize = 12

fig3, ax3 = plt.subplots(figsize=(8, 8))
plt.suptitle(f'    Mean Distribution of Emission Metrics \n for {sort_plot3}: {subcat_plot3_sel}  \n (2013 to 2023)',y=0.53,size=18)

outer_circle = plt.pie(em_percent,labels=categories,colors=category_colors,radius=1,labeldistance=1.05,
               autopct='%1.1f%%',pctdistance=0.875,textprops={'fontsize':fontsize})

center_circle = plt.Circle((0, 0), 0.75, color='white', fc='white', linewidth=1.25)
fig3 = plt.gcf()
fig3.gca().add_artist(center_circle)
ax3.axis('equal');

st.pyplot(fig3)

### Plot-4
st.subheader('Aggregate Emission Metrics Across All Companies: 2013 to 2023')

country_sel_plot4 = st.selectbox('Select the Country to sort by',country_names,key='country_sel_plot4')

font = {'size':16}
plt.rc('font',**font)

rows = int(np.round(len(data.columns)/2))
cols = 2
fig4,ax4 = plt.subplots(rows,cols,figsize=(20,5*rows))

plt.suptitle(f'Aggregate Emission Metrics Across All Companies in {country_sel_plot4}: 2013 to 2023',y=1,size=20)

if country_sel_plot4 == 'All Countries':
    for i,col in enumerate(data.columns):
        data.groupby('year')[col].sum().plot(kind='bar',ax=ax4.ravel()[i],title=col+': '+emission_metrics.loc[col,'Name'])
        ax4.ravel()[i].tick_params(axis='both',labelsize=14)
else:
    for i,col in enumerate(data.columns):
        data.groupby(['country','year'])[col].mean()[country_sel_plot4].plot(kind='bar',ax=ax4.ravel()[i],title=col+': '+emission_metrics.loc[col,'Name'])
        ax4.ravel()[i].tick_params(axis='both',labelsize=14)
    
plt.tight_layout()
st.pyplot(fig4)

### Plot-5
st.subheader('Emission Metrics for Companies between 2013 and 2023')

metric_sel_plot5 = st.selectbox('Select the emission metric you want to assess',em_metric_names,key='metric_sel_plot5')
company_sel_plot5 = st.selectbox('Select the company to assess',company_names,key='company_sel_plot5')

fig5 = plt.figure(figsize=(12,5))
data.loc[company_sel_plot5][metric_sel_plot5].plot(kind='bar')
plt.title(f'{metric_sel_plot5} Metric for Company: {company_sel_plot5} between 2013 and 2023',pad=10,size=16)
plt.xlabel('Year',labelpad=10,size=14)
plt.xticks(range(len(years)),years)
plt.tick_params(axis='both',labelsize=12);

st.pyplot(fig5)

### Plot-6
st.subheader('Year on Year Change in Emission Metric for Country or Company between 2013 and 2023')

sort_plot6 = st.selectbox('Select an option to sort by',['Country','Company'],key='sort_plot6')

if sort_plot6 == 'Country':
    sort_plot6_sel = 'country'
    subcat_plot6 = data.index.levels[1].to_list()
else:
    sort_plot6_sel = 'isin'
    subcat_plot6 = data.index.levels[0].to_list()   

subcat_plot6_sel = st.selectbox('Select the sub-category',subcat_plot6,key='subcat_plot6_sel')
em_metric_sel_plot6 = st.selectbox('Select the emission metric to assess',em_metric_names,key='em_metric_sel_plot6')

fig6 = plt.figure(figsize=(20,5))
plt.title(f'Year on Year change in {em_metric_sel_plot6} Emission for {sort_plot6}: {subcat_plot6_sel} Between 2013 and 2023',pad=10,size=18)
(data.groupby([sort_plot6_sel,'year'])[em_metric_sel_plot6].mean().pct_change()*100)[subcat_plot6_sel].drop(2013,axis=0).plot(kind='bar')
plt.xlabel('Year',labelpad=10,size=16)
plt.ylabel('Percent Change (%)',labelpad=10,size=16);

st.pyplot(fig6)
