from sklearn.preprocessing import MinMaxScaler
import pandas
import numpy as np

class PrepData():
    def scale(self, diff_data):
        scaler = MinMaxScaler(feature_range=(-1,1))
        #diff_data = diff_data.to_numpy()
        for i in diff_data.values:
            i.reshape(i.shape[0], 1)
        values = diff_data.values
        for i in range(len(values)):
            values[i] = values[i].reshape(values[i].shape[0], 1)
            values[i] = scaler.fit_transform(values[i]);
        #diff_data.values.reshape(diff_data.shape[0],1)
        #diff_data.reshape(diff_data.shape[0], 4)
        #diff_data = scaler.fit_transform(values);
        #diff_data = diff_data.reshape(diff_data.shape[0])
        return values
        
    def difference(self, data, interval=1):
        diff = list()
        for i in range(interval, len(data)):
            value = data[i] - data[i - interval]
            diff.append(value)
        return pandas.Series(diff)

    def invert_Diff(history, value, interval=1):
        return value+history[-interval]
        
    def prep_Data(self, data):
       values = data.iloc[:].values
       diff_data = self.difference(values)
       scaled_data = self.scale(diff_data)
       data.iloc[:190] = scaled_data
       return data