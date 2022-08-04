from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import pandas
import numpy as np


class PrepData():
    def scale(self, data):
        global scaler
        scaler = MinMaxScaler(feature_range=(0,1))
        values = scaler.fit_transform(data.values)
        data = pandas.DataFrame(values,columns=data.columns,index=data.index)
        return data

    def normalize(self, data):
        d_len = len(data)
        train_len = round(d_len*.7)
        val_len = round(d_len*.9)
        test_len = round(d_len)
        data_list = [data[:train_len], data[train_len:val_len], data[val_len:test_len]]
        global undo_n
        undo_n = []
        for i in data_list:
            for x in i.columns:
                std = np.std(i[x].values)
                mean = np.mean(i[x].values, dtype=np.float64)
                values = [(y-mean)/std for y in i[x].values]
                i[x].iloc[:] = values
            undo_n = np.append(undo_n, np.array([std, mean]))
        undo_n = undo_n.reshape(len(data_list), 2)

        return data_list[0], data_list[1], data_list[2]

    def denormalize(self, data):      
        
        std = undo_n[-1,0] 
        mean = undo_n[-1, 1]
        v_arr = data.numpy()
        for i in range(v_arr.shape[1]):
            values = [(y*std) + mean for y in v_arr[i]]
            v_arr[i] = values
        return data[0], data[1], data[2]
           
    def prep_data(self, data):
       train_data, val_data, test_data = self.normalize(data)     
       return train_data, val_data, test_data

    def revert_data(self, data):
       train_data, val_data, test_data = self.denormalize(data)
       return train_data, val_data, test_data