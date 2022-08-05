import os
import numpy as np
import numpy
import model.WindowGen as win
import pandas


cd = os.getcwd()
path = cd + "/Google.csv"


def Predict():
    names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    dataframe = pandas.read_csv(path, header=None, names=names, parse_dates=[0], dtype=numpy.float64, index_col=0, skiprows=1)
    dataframe = dataframe.iloc[:,[0]].astype(numpy.float64)
    window = win.temp.WindowGen(input_width=100, label_width=25, shift=25, data=dataframe)
    feedback_model = win.temp.FeedBack(units=16, out_steps=25)
    history, batch_size = win.temp.compile_and_fit(feedback_model, window)
    plot_url = window.plot(feedback_model)
    index = history.history['val_loss'].index(np.min(history.history['val_loss']))
    t_loss = history.history['loss'][index]
    t_rmse = history.history['root_mean_squared_error'][index]
    v_rmse = history.history['val_root_mean_squared_error'][index]
    v_loss = history.history['val_loss'][index]
    epoch_num = len(history.epoch)

    return plot_url, t_loss, t_rmse, v_loss, v_rmse, epoch_num, batch_size





