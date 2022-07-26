import os
import numpy
import pandas
import json
from sklearn.preprocessing import StandardScaler

def scaledData():
    cd = os.getcwd()
    path = cd + "/Google.csv"
    names = ['date', 'high', 'low', 'close', 'adjclose', 'volume' ]
    dataframe = pandas.read_csv(path, names = names)
    array = dataframe.values

    Input = array[:,0:5]
    y = array[:,5]

    scaler = StandardScaler().fit(Input)
    rescaledInput = scaler.transform(Input)

    return json.dumps(rescaledInput[0:5,:])
