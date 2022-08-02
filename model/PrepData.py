from sklearn.preprocessing import MinMaxScaler
import pandas
import numpy as np

class PrepData():
    def scale(self, data):
        scaler = MinMaxScaler(feature_range=(-1,1))
        values = scaler.fit_transform(data.values)
        data=pandas.DataFrame(values,columns=data.columns,index=data.index)
        return data, scaler
        
    def prep_data(self, data):
       data, scaler = self.scale(data)      
       return data, scaler

    def revert_data(self, data, scaler):
        return scaler.inverse_transform(data)