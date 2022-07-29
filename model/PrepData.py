from sklearn.preprocessing import MinMaxScaler
import pandas
class PrepData():
    def scale(self, diff_data):
        scaler = MinMaxScaler(feature_range=(-1,1))
        diff_data = diff_data.values.reshape(diff_data.shape[0], 1)
        diff_data = scaler.fit_transform(diff_data);
        diff_data = diff_data.reshape(diff_data.shape[0])
        return diff_data
        
    def difference(self, data, interval=1):
        diff = list()
        for i in range(interval, len(data)):
            value = data[i] - data[i - interval]
            diff.append(value)
        return pandas.Series(diff)

    def invert_Diff(history, value, interval=1):
        return value+history[-interval]
        
    def prep_Data(self, data):
       values = data.iloc[:,0].values
       diff_data = self.difference(values)
       scaled_data = self.scale(diff_data)
       data.iloc[:140,0] = scaled_data
       return data