from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas
import numpy as np
import math


class PrepData():
    def normalize(self, data):
        d_len = len(data)
        train_len = round(d_len*.7)
        val_len = round(d_len*.9)
        global std
        global mean
        mean = np.mean(data)
        std = np.std(data)
        data.iloc[:] = [(x-mean)/std for x in data.values] 
        train_data = data[:train_len]
        val_data = data[train_len:val_len]
        test_data = data[val_len:]
        return train_data, val_data, test_data

    def denormalize(self, data):             
        v_arr = data.numpy()
        for i in range(v_arr.shape[1]):
            values = [(y*std) + mean for y in v_arr[:, i]]
            v_arr[:, i] = values
        return v_arr

    def scale(self, data):
        global scaler
        scaler = MinMaxScaler(feature_range=(0,1))
        d_len = len(data)
        train_len = round(d_len*.7)
        val_len = round(d_len*.9)
        data_list = [data[:train_len], data[train_len:val_len], data[val_len:]]
        for i in range(len(data_list)):                         
            data_list[i] = pandas.DataFrame(data=scaler.fit_transform(data_list[i]), columns=data_list[i].columns, index=data_list[i].index)
        return data_list[0], data_list[1], data_list[2]

    def invert_scale(self, data):
        return scaler.inverse_transform(data)
           
    def prep_data(self, data):
       train_data, val_data, test_data = self.normalize(data)     
       #train_data, val_data, test_data = self.scale(data) 
       return train_data, val_data, test_data

    def revert_data(self, data):
       data = self.denormalize(data)
       #data = self.invert_scale(data)
       return data