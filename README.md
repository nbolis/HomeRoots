# Home Roots
## The Urban Farm Companion Tool

Home Roots uses urban farm crop data together with weather data to recommend which crops to plant in a given location and at a given month, based on what urban farmers in locations with similar weather plant. 

#### Index
1. In "CropDataCleaning_HR1.ipynb" I clean the urban farm data and construct the target variable to be predicted. The target variable is the number of a particular crop planted at a location and month weighted by total number of crops in planted in that location.
2. In "WeatherDataClean_HR2.ipynb" I average the weather data over one month periods. The weather data will be used as the features of the model. 
3. In "WeatherCropMerge_HR3.ipynb" I merge the crop number target variable with the weather data for each location and date. 
4. "HomeRoots.py" contains the class with both linear and Random Forest regression models which I compare in "RunHomeRoots_HR4.ipynb". 
5. "Streamlit_HR.ipynb" contains the Streamlit deployed HomeRoots app. 