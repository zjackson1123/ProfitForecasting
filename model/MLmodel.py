import os
import base64
from turtle import numinput
import numpy
import pandas
from io import BytesIO
import matplotlib.pyplot as plot
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures as pf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from math import sqrt

cd = os.getcwd()
path = cd + "/Google.csv"
MSE = 0

def parser(x):
    dates = pandas.to_datetime(x)
    return dates.strftime("%Y%m%d")

def shiftTimeSeries(data, numInput, numDays):
    df = pandas.DataFrame(data)
    n_vars = 1 if type(data) is list else data.shape[1]
    columns, names = list(), list()
    for i in range(numInput, 0, -1):
        columns.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, numDays):
        columns.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    agg = pandas.concat(columns, axis=1)
    agg.columns = names
    agg.dropna(inplace=True)
    return agg
    


def scale(data):
    scaler = MinMaxScaler(feature_range=(-1,1))
    data_scaled = scaler.fit_transform(data)
    data_scaled = data_scaled.reshape(len(data_scaled),1)
    return scaler, data_scaled

def invertScale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.invert_transform(array)
    return inverted[0,-1]

def diff(dataset, interval=1):
    diff=list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return pandas.Series(diff)

def invertDiff(history, yhat, interval=1):
    return yhat+history[:41][-interval]

def fit_ltsm(train, numInput, numDays, batch_size, nb_epoch, neurons):
    x, y = train[:,0:numInput], train[:,numInput:]
    x = x.reshape(x.shape[0], 1, x.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, x.shape[1], x.shape[2]), stateful=True))
    model.add(Dense(y.shape[1]))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(x, y, epochs = 1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model

def forecast_lstm(model, batch_size, x):
    x = x.reshape(1, 1, len(x))
    yhat = model.predict(x, batch_size=batch_size)
    return yhat[0,0]

def supervisedLearning():
    numInput = 1
    numDays = 30
    dataframe = pandas.read_csv(path, header=0, parse_dates=[0], index_col=0, date_parser=parser, skiprows=1)
    dates = dataframe.iloc[91:121].index
    raw_values = dataframe.iloc[:121,0].values
    difference = diff(raw_values, 1)
    diff_values = difference.values
    diff_values = diff_values.reshape(len(diff_values), 1)
    scaler, scaled_values = scale(diff_values)
    supervised = shiftTimeSeries(scaled_values, numInput, numDays)
    supervised_values = supervised.values
    train, test = supervised_values[:90], supervised_values[91:] 
    lstm_model = fit_ltsm(train, numInput, numDays, 1, 100, 4)
    train_reshaped = train[:,0].reshape(len(train), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=1)

    predictions = list()
    for i in range(len(test)):
        x, y = test[i, 0:-1], test[i, -1]
        yhat = forecast_lstm(lstm_model, 1, x)
        yhat = invertDiff(raw_values, yhat, len(test) + 1 - i)
        predictions.append(yhat)
        expected = raw_values[len(train) + i + 1]

    global MSE 
    MSE = sqrt(mean_squared_error(raw_values[31:41], predictions))
    plot.plot(dates, raw_values[31:41], color='r')
    plot.plot(dates, predictions, color='b')
    img = BytesIO()
    plot.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
    return plot_url


#def scaledData():
#    np = dataframe.to_numpy()
#    y = np[:,4].reshape(-1,1)
#    x = np[:,1].reshape(-1,1)
#    scaler = MinMaxScaler(feature_range=(0, 1))
#    plot.plot(y, x)  
#    lm = LinearRegression()
#    lm.fit(y, x)
#    y_pred = lm.predict(y)
#    plot.plot(y, y_pred, color='r')
#    poly = pf(degree=5, include_bias=False)
#    poly_features = poly.fit_transform(y)
#    poly_reg_model = LinearRegression()
#    poly_reg_model.fit(poly_features, x)
#    y_pred_poly = poly_reg_model.predict(poly_features)
#    plot.plot(y, y_pred_poly, color='g')
#    img = BytesIO()
#    plot.savefig(img, format='png')
#    plot.close()
#    img.seek(0)
#    plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
#    return plot_url
    #return dataframe

    #def testModel():
#    test_y = dataframe.iloc[31:61,1].values
#    train_y = dataframe.iloc[:30,1].values
#    test_x = dataframe.iloc[31:61,0]
#    train_x = dataframe.iloc[:30,0]
#    y = arr[:,4]
#    x = arr[:,1]
#    history = [x for x in train_y]
#    predictions = list()
#    for i in range(len(test_y)):
#        predictions.append(history[-1])
#        history.append(test_y[i])
#    global MSE 
#    MSE = sqrt(mean_squared_error(test_y, predictions))
#    plot.plot(test_x, test_y, color='b')
#    plot.plot(test_x,predictions, color='orange')
#    img = BytesIO()
#    plot.savefig(img, format='png')
#    img.seek(0)
#    plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
#    return plot_url


