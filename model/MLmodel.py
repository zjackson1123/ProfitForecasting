import os
import tensorflow as tf
import numpy
import Model.WindowGen as win
import Model.PrepData as pd
import pandas
from tensorflow import keras
from tensorflow.keras import layers


cd = os.getcwd()
path = cd + "/Google.csv"


def Predict():
    names = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    dataframe = pandas.read_csv(path, header=None, names=names, parse_dates=[0], dtype=numpy.float64, index_col=0, skiprows=1)
    dataframe = dataframe.iloc[:,[0,3]].astype(numpy.float64)
    prep = pd.PrepData()
    prepped_data, scaler = prep.prep_data(dataframe.iloc[:1001])
    train_df = prepped_data[:499]
    val_df = prepped_data[500:550]
    test_df = prepped_data[551:650]
    window = win.temp.WindowGen(input_width=100, label_width=25, shift=25, train_df = train_df, val_df = val_df, test_df = test_df)
    feedback_model = win.temp.FeedBack(units=100, out_steps=25)
    history = win.temp.compile_and_fit(feedback_model, window)
    plot_url = window.plot(scaler, feedback_model)

    return plot_url





