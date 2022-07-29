import os
import base64
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

def shiftTimeSeries(data, lag=1):
    df = pandas.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pandas.concat(columns, axis=1)
    df.fillna(0,inplace=True)
    return df

def scale(train, test):
    scaler = MinMaxScaler(feature_range=(-1,1))
    scaler = scaler.fit(train)
    train = train.reshape(train.shape[0],train.shape[1])
    train_scaled = scaler.transform(train)
    test = test.reshape(test.shape[0], test.shape[1])
    test_scaled = scaler.transform(test)
    return scaler, train_scaled, test_scaled

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
    return yhat+history[:121][-interval]

def fit_ltsm(train, batch_size, nb_epoch, neurons):
    x, y = train[:,0:-1], train[:,-1]
    x = x.reshape(x.shape[0], 1, x.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, batch_input_shape=(batch_size, x.shape[1], x.shape[2]), stateful=True))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    for i in range(nb_epoch):
        model.fit(x, y, epochs = 1, batch_size=batch_size, verbose=0, shuffle=False)
        model.reset_states()
    return model

def forecast_lstm(model, batch_size, x):
    x = x.reshape(1, 1, len(x))
    yhat = model.predict(x, batch_size=batch_size)
    return yhat[0,0]

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

def supervisedLearning(epochs, dataframe, plot_urls):

    dates = dataframe.iloc[91:121].index
    raw_values = dataframe.iloc[:,0].values
    diff_values = diff(raw_values, 1)
    supervised = shiftTimeSeries(diff_values, 1)
    supervised_values = supervised.values
    train, test = supervised_values[:90], supervised_values[91:121] 
    
    scaler, train_scaled, test_scaled = scale(train, test)
   
    lstm_model = fit_ltsm(train_scaled, 1, epochs, 4)
    train_reshaped = train_scaled[:,0].reshape(len(train_scaled), 1, 1)
    lstm_model.predict(train_reshaped, batch_size=1)
    error = list()
    predictions = list()
    for i in range(len(test_scaled)):
        x, y = test_scaled[i, 0:-1], test_scaled[i, -1]
        yhat = forecast_lstm(lstm_model, 1, x)
        yhat = invertDiff(raw_values, yhat, len(test_scaled) + 1 - i)
        predictions.append(yhat)
        expected = raw_values[len(train) + i + 1]

    mse = sqrt(mean_squared_error(raw_values[91:121], predictions))
    print('%d) Test RMSE: %.3f' % (1, mse))
    error.append(mse)
    plot.plot(raw_values[91:121], color='r')
    plot.plot(predictions, color='b')
    img = BytesIO()
    plot.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
    plot_urls.append(plot_url)
    return error


def errorScores():
     dataframe = pandas.read_csv(path, header=0, parse_dates=[0], index_col=0, date_parser=parser, skiprows=1)
     errors = pandas.DataFrame()
     plot_urls = list()
     epochs = [10, 50, 100, 200, 300]
     for e in epochs:         
         errors[str(e)] = supervisedLearning(e, dataframe, plot_urls)


     return plot_urls, errors.to_html

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



