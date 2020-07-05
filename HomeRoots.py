import pandas as pd
import numpy as np
import datetime
import os
import json 
import pickle 

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import MinMaxScaler

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

from sklearn.model_selection import cross_validate
from sklearn.model_selection import KFold
from sklearn.model_selection import GroupKFold

from sklearn import metrics

class HomeRoots:
    """
   Home Roots model exploration and deployment. 
   In this module:
   
   - import merged and cleaned weather-crop data, 
   - scale data with SKlearn scaler
   - split data into training/test sets in two ways:
           - location dependent
           - location indipendent
   -  Train two models:
           - Linear
           - Random Forest
   - Group K-Fold cross validation for parameter optimization 
    """
    weather_features = ['mintempC', 'maxtempC', 'sunHour', 'cloudcover', 'humidity', 
                        'precipMM', 'pressure', 'windspeedKmph']
    
    with open('../data/merged_data/veg_dict.json') as json_file:
        datafile_veggie_dict = json.load(json_file)
        
    with open('../data/merged_data/fru_dict.json') as json_file:
        datafile_fruit_dict = json.load(json_file)
        
    with open('../data/merged_data/her_dict.json') as json_file:
        datafile_herb_dict = json.load(json_file)
        
    with open('../data/merged_data/flo_dict.json') as json_file:
        datafile_flower_dict = json.load(json_file)

    datafile_veggie_dict.update(datafile_fruit_dict)
    datafile_veggie_dict.update(datafile_herb_dict)
    datafile_veggie_dict.update(datafile_flower_dict)
        
        
    def __init__(self, crop, **kwargs):
        self.crop = crop
        self.datafile = self.datafile_veggie_dict[crop]
        self.split   = kwargs.get('split', 'location_dependent')
        self.percent = kwargs.get('percent', 0.3)
        self.is_scaled = kwargs.get('scaled', True)
        ## Setup the data
        self._setup_the_data_()
        if self.is_scaled:
            self._scaler_()
        else:
            self._unscaler_()
        ## Split-by choice
        self._prepare_model_()
        
        
    def _setup_the_data_(self):
        ## Read in the given data file
        self.df = pd.read_csv(self.datafile)
    
    def _crop_weather_feature_(self):
        subdf = self.df[ self.df['crop'] == self.crop ]
        subdf = subdf[['quant_per_tot', 'mintempC', 'maxtempC', 
                       'sunHour', 'cloudcover', 'humidity', 'precipMM', 
                       'pressure', 'windspeedKmph']]
        return subdf
    
    def zip_frame(self):
        return self.df['zip']
    
    def _scaler_(self):
        subdf = self._crop_weather_feature_()
        self.scaler_tot =  MinMaxScaler()
        self.scaler_tot.fit(subdf)
        scaled_cropwea = self.scaler_tot.transform(subdf)
        feats_cropwea = subdf.columns.to_numpy().tolist()
        scaled_crop_wea = pd.DataFrame(data = scaled_cropwea, columns = feats_cropwea)  
        scaled_crop_wea['zip'] = self.zip_frame()
        self.hrdata = scaled_crop_wea
        return scaled_crop_wea
    
    def _unscaler_(self):
        subdf = self._crop_weather_feature_()
        unscaled_cropwea = subdf
        feats_cropwea = subdf.columns.to_numpy().tolist()
        unscaled_crop_wea = pd.DataFrame(data = unscaled_cropwea, columns = feats_cropwea)  
        unscaled_crop_wea['zip'] = self.zip_frame()
        self.hrdata = unscaled_crop_wea
        return unscaled_crop_wea
    
    def _prepare_model_(self):
            
        if self.split == 'location_dependent':
            thirty_percent_data = self.hrdata.shape[0]*self.percent
            number_unique_loc= self.hrdata['zip'].unique().shape[0]
            number_time_steps_per_loc = self.hrdata.shape[0]/number_unique_loc
            number_loc_in_thirty_percent = int(thirty_percent_data/number_time_steps_per_loc)
            unique_loc = self.hrdata['zip'].unique()
            np.random.seed(3)
            np.random.shuffle(unique_loc)
            
            test_loc = unique_loc[0:number_loc_in_thirty_percent].tolist()
            scaled_crop_wea_loc = self.hrdata.set_index('zip')
            
            train = scaled_crop_wea_loc.drop(test_loc, axis=0)
            self.test = scaled_crop_wea_loc.loc[test_loc]
            
            self.groups = train.index.to_numpy()
            self.n_sp = np.unique(self.groups).shape[0]
            X_train = train[self.weather_features].to_numpy()
            X_test  = self.test[self.weather_features].to_numpy()
            Y_train = train['quant_per_tot'].to_numpy()
            Y_test  = self.test['quant_per_tot'].to_numpy()
            self.X_train = X_train
            self.X_test  = X_test
            self.Y_train = Y_train
            self.Y_test  = Y_test
        
        elif self.split == 'location_independent':
            X = self.hrdata[self.weather_features].to_numpy()
            Y = self.hrdata['quant_per_tot'].to_numpy()
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=self.percent, random_state = 3)
            self.X_train = X_train
            self.X_test  = X_test
            self.Y_train = Y_train
            self.Y_test  = Y_test
        else:
            raise(NameError) 
            

    def linear_model(self, one_spot = None):
        self.lin = LinearRegression().fit(self.X_train, self.Y_train)
        print(self.Y_train.shape)
        y_pred = self.lin.predict(self.X_test)
        
        df_pred = pd.DataFrame({'Actual': self.Y_test.flatten(), 'Predicted': y_pred.flatten()})
        if one_spot is None:
            one_spot_predict = None
        else:
            one_spot_scaled = self.scaler_tot.transform(one_spot)
            one_spot_scaled = one_spot_scaled[:, :8]
            one_spot_predict = self.lin.predict(one_spot_scaled)
            
        mae =  metrics.mean_absolute_error(self.Y_test, y_pred) 
        mse =  metrics.mean_squared_error(self.Y_test, y_pred) 
        rmse =  np.sqrt(metrics.mean_squared_error(self.Y_test, y_pred))
        print('Mean Absolute Error:', mae)  
        print('Mean Squared Error:', mse)  
        print('Root Mean Squared Error:', rmse)
        return df_pred, mae, mse, rmse, one_spot_predict
    
    def random_forest_model(self, n_est, max_d, one_spot = None):
        self.rf = RandomForestRegressor(n_estimators = n_est,
                          max_depth = max_d)
        print(self.Y_train.shape)
        self.rf.fit(self.X_train, self.Y_train)
        cwd = os.path.abspath(f'rf_model_{self.crop}_n{n_est}_m{max_d}.pkl')
        pickle.dump(self.rf, open(cwd, 'wb'))
        RF_predictions = self.rf.predict(self.X_test)
        if one_spot is None:
            one_spot_predict = None 
        else:
            if self.is_scaled ==True:
                one_spot_scaled = self.scaler_tot.transform(one_spot)[:, :8]
            else:
                print(one_spot[:, 8:])
                one_spot_scaled = one_spot[:, :8]
            one_spot_predict = self.rf.predict(one_spot_scaled)
      
        df_pred = pd.DataFrame({'Actual': self.Y_test.flatten(), 'Predicted': RF_predictions.flatten()})
        mae = metrics.mean_absolute_error(self.Y_test, RF_predictions)
        mse = metrics.mean_squared_error(self.Y_test, RF_predictions)
        rmse = np.sqrt(metrics.mean_squared_error(self.Y_test, RF_predictions))
        print('Mean Absolute Error:', mae)  
        print('Mean Squared Error:', mse)  
        print('Root Mean Squared Error:', rmse)
        return df_pred, mae, mse, rmse, one_spot_predict
    
    
    def group_cv(self, model, n_estim=None, max_d=None):
        X = self.X_train
        y = self.Y_train
        gkf = GroupKFold(n_splits=self.n_sp)
        mae_avg_cv = []
        mse_avg_cv = []
        rmse_avg_cv = []
        
        
        for train, test_cv in gkf.split(X, y, groups=self.groups):
            X_train, X_test_cv = X[train], X[test_cv]
            y_train, y_test_cv = y[train], y[test_cv]
            
            if model == 'RF':
                rf = RandomForestRegressor(n_estimators = n_estim, max_depth = max_d)
                rf.fit(X_train, y_train)
                RF_predictions_cv = rf.predict(X_test_cv)
                mae_cv = metrics.mean_absolute_error(y_test_cv, RF_predictions_cv)
                mse_cv = metrics.mean_squared_error(y_test_cv, RF_predictions_cv)
                rmse_cv = np.sqrt(metrics.mean_squared_error(y_test_cv, RF_predictions_cv))
 
            if model == 'lin':
                lin = LinearRegression().fit(X_train, y_train)
                y_pred_cv = lin.predict(X_test_cv)
                mae_cv = metrics.mean_absolute_error(y_test_cv, y_pred_cv)
                mse_cv = metrics.mean_squared_error(y_test_cv, y_pred_cv)
                rmse_cv = np.sqrt(metrics.mean_squared_error(y_test_cv, y_pred_cv))

      
            mae_avg_cv.append(mae_cv)
            mse_avg_cv.append(mse_cv)
            rmse_avg_cv.append(rmse_cv)
        Avg_mae_cv = np.asarray(mae_avg_cv).mean()
        Avg_mse_cv = np.asarray(mse_avg_cv).mean()
        return mae_avg_cv, mse_avg_cv, rmse_avg_cv, Avg_mse_cv, Avg_mae_cv 

