import streamlit as st
import pandas as pd
from typing import List, Optional
import pickle
import markdown
import os
import numpy as np
#from HomeRoots import date_month_appender, month_avger

from wwo_hist import retrieve_hist_data
from datetime import date




import matplotlib.pyplot as plt
from matplotlib import colors, cm
import squarify    # pip install squarify (algorithm for treemap)

from PIL import Image


###### LOGO 
#with open("style.css") as f:
 #   st.markdown('<style>{}</style>'.format(f.read()), unsafe_allow_html=True)

image = Image.open('HomeRoots_logoTs.png')   #('../assets/HomeRoots_logoTs.png')
#with st.center():  
st.image(image, format='PNG')


###### user input 
#st.text("")
st.text("")
st.text("")
st.markdown('## **What are the best crops to plant in my Urban Farm?**')
    
user_input = st.text_input("Enter Zip Code", 0)

month_choice = st.selectbox("Select Month to Plant", options=["January", "Febuary", "March", "April", "May",
                                               "June", "July", "August", "September", "October",
                                               "November", "December"])   

crop_choice = st.selectbox("Select Crop Category", options=["Vegetables", "Fruit", "Herbs", "Flowers"])  


## Features needed 
weather_features = ['mintempC', 'maxtempC', 'sunHour', 'cloudcover', 'humidity', 
                        'precipMM', 'pressure', 'windspeedKmph'] 

dict_month = {"January":["1-JAN","31-JAN"], "Febuary": ["1-FEB", "28-FEB"], "March": ["1-MAR", "31-MAR"], "April":["1-APR", "30-APR"], "May":["1-MAY","31-MAY"], "June": ["1-JUN","30-JUN"], "July":["1-JUL","31-JUL"], "August":["1-AUG","31-AUG"], "September":["1-SEP","30-SEP"],"October":["1-OCT", "31-OCT"], "November":["1-NOV","30-NOV"], "December": ["1-DEC", "31-DEC"]}
c_year = int(date.today().year)-1


#### Weather API


if user_input =='0':
    ## Import default data
    datafortest = pd.read_csv('cropweatomato.csv')    #('../data/merged_data/cropweatomato.csv')
    data_to_use = datafortest.iloc[int(user_input)][weather_features].to_numpy().reshape(1,-1)
    
else:
    frequency=24
    start_date = f'{dict_month[month_choice][0]}-{c_year}'  #'1-MAY-2020'
    end_date = f'{dict_month[month_choice][1]}-{c_year}'   #'30-MAY-2020'
    api_key = '0c08bdf5a0fd4090a2e180659201006'
    #api_key = '6fbf5e0f161847788fb170515201006'
    #api_key = '7d5b9f646cca4e46aef152315201006' 
    #api_key = 'bcc6f5e78a0e45fe9ba60011201006'
    #api_key = '6fcff801e794455cb7550335201006'
    #api_key = 'a98b3f03217d44f094235813201006'
    location_list = [user_input]    

    hist_weather_data = retrieve_hist_data(api_key, location_list, start_date,
                                           end_date,frequency, location_label = False,
                                           export_csv = True, store_df = True)

    ## get downloaded weather data
    #name_of_s = [os.path.join(root_path, f) for f in s_in_directory if f.endswith('.csv')]
    iname = f'{user_input}.csv'
    input_data = pd.read_csv(iname)
  
    ## average weather data for month
    
    arr =input_data[weather_features]
    data_to_use = arr.mean().to_numpy().reshape(1,-1)
  
    
    ## delete weather data file
    os.remove(iname)

###### end weather data part #######        
        



### Import pickles ###

veggies = {}
flist = [f for f in os.listdir() if f.endswith('.pkl')]
for f in flist:
  with open(os.path.abspath(f), 'rb') as ff:
    ## We expect everything of the form rf_model_veggie_n_m
    filestring = (f.split('.')[0]).split('_')
    if len(filestring) != 5:
        continue
    veggies[ filestring[2] ] = pickle.load(ff)
    


## Crop catedories ## 'tomato'

vegetable = ['amaranth', 'arugula','asparagus', 'peppers',  
             'beans','beetroot', 'peas', 'bok choy', 'gourd',
             'broccoli', 'fennel','brussels sprouts', 'squash', 'cabbage',
             'carrots', 'celery', 'chard' , 'collards', 'corn', 'cucumbers',
             'kale', 'eggplant','garlic', 
             'lettuce', 'radish', 'jalapenos', 'kohlrabi', 'leeks','okra', 
             'onion','potatoes', 'pumpkin', 'spinach', 'sweet potato', 'taro', 'tomatillos', 
             'turnip','zucchini', 'artichoke', 'callaloo' ]

def rf_predict(test_data):
    rf_dict_dep = {}
    for veg in vegetable:
        RF_model = veggies[veg]
        one_spot_predict = RF_model.predict(test_data)
        rf_dict_dep[veg] = one_spot_predict  #[-1][0], one_spot_predict[2]]
     
    return rf_dict_dep

dict4 = rf_predict(data_to_use)

df = pd.DataFrame(data = dict4, index = ['Planting Score']).T.sort_values('Planting Score',ascending = False)

cmap = cm.RdYlGn
mini=min(df['Planting Score'])
maxi=max(df['Planting Score'])
norm = colors.Normalize(vmin=mini, vmax=maxi)
colors = [cmap(norm(value)) for value in df['Planting Score']]

st.write('')
st.write('')
st.markdown('**Top Crops Field-View:** The larger the field-square, the better fit a crop is for your location and chosen planting month.')


fig = plt.gcf()
ax = fig.add_subplot()
fig.set_size_inches(16, 9)
dfindex = df.index

plt.rc('font', size=10)


le_axis = squarify.plot(sizes=df['Planting Score'], label=dfindex,
                        alpha=.7, color = colors )
#ax.patch.set_facecolor('#ababab')
#ax.patch.set_alpha(0.5)
le_axis.invert_yaxis()
plt.axis('off')
st.pyplot()

#st.write(df.head().T)
st.markdown('### In locations with similar climate as yours:')

st.markdown(f' - {df.index[0]} is planted {np.floor(df.iloc[0][0]*100)}% of the time')

st.markdown(f' - {df.index[1]} is planted {np.floor(df.iloc[1][0]*100)}% of the time')
#st.write(f'In locations wit

st.markdown(f' - {df.index[2]} is planted {np.floor(df.iloc[2][0]*100)}% of the time')

st.markdown(f' - {df.index[3]} is planted {np.floor(df.iloc[3][0]*100)}% of the time')

st.markdown(f' - {df.index[4]} is planted {np.floor(df.iloc[4][0]*100)}% of the time')



#st.write(f'In locations wit
#st.write(f'In locations with similar climate as yours {df.index[0]} are planted {np.floor(df.iloc[0][0]*100)}% of the time')
