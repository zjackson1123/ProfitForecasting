import os
import base64
from io import BytesIO
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
import numpy
import pandas
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.preprocessing import MinMaxScaler

cd = os.getcwd()
path = cd + "/Google.csv"
names = ['dates', 'open', 'high', 'low', 'close', 'adjclose', 'volume' ]
dataframe = pandas.read_csv(path, names=names, skiprows=1)

def trainModel():
    arr =  dataframe.iloc[:3280,1:2].values
    y = arr[:,4]


def scaledData():
    np = dataframe.to_numpy()
    y = np[:,4].reshape(-1,1)
    x = np[:,1].reshape(-1,1)
    scaler = MinMaxScaler(feature_range=(0, 1))
    plot.plot(y, x)  
    lm = LinearRegression()
    lm.fit(y, x)
    y_pred = lm.predict(y)
    plot.plot(y, y_pred, color='r')
    poly = pf(degree=5, include_bias=False)
    poly_features = poly.fit_transform(y)
    poly_reg_model = LinearRegression()
    poly_reg_model.fit(poly_features, x)
    y_pred_poly = poly_reg_model.predict(poly_features)
    plot.plot(y, y_pred_poly, color='g')
    img = BytesIO()
    plot.savefig(img, format='png')
    plot.close()
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
    return plot_url
    #return dataframe



