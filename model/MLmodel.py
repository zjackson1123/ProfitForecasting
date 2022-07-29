import os
import tensorflow as tf
import numpy
from Model import WindowGen
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
    dataframe = dataframe.iloc[:,0:4].astype(numpy.float64)
    prep = pd.PrepData()
    prepped_data = prep.prep_Data(dataframe.iloc[:141])
    train_df = prepped_data[:90]
    val_df = prepped_data[90:110]
    test_df = prepped_data[110:140]
    window = win.WindowGen(input_width=90, label_width=1, shift=1, train_df = train_df, val_df = val_df, test_df = test_df)
    example_window = train_df.iloc[:,0].values
    example_inputs, example_labels = window.split_window(example_window)

    print('All shapes are: (batch, time, features)')
    print(f'Window shape: {example_window.shape}')
    print(f'Inputs shape: {example_inputs.shape}')
    print(f'Labels shape: {example_labels.shape}')
    plot_url = window.plot()
    return plot_url





