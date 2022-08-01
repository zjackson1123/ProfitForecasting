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
    dataframe = dataframe.iloc[:,0:4].astype(numpy.float64)
    prep = pd.PrepData()
    prepped_data = prep.prep_Data(dataframe.iloc[:191])
    train_df = prepped_data[:89]
    val_df = prepped_data[90:140]
    test_df = prepped_data[141:190]
    window = win.temp.WindowGen(input_width=24, label_width=24, shift=1, label_columns=['Open'], train_df = train_df, val_df = val_df, test_df = test_df)
    baseline = win.temp.Baseline(label_index=window.column_indices['Open'])
    baseline.compile(loss=tf.keras.losses.MeanSquaredError(), metrics=[tf.keras.metrics.MeanAbsoluteError()])
    linear = tf.keras.Sequential([
        tf.keras.layers.Dense(units=1)
    ])
    history = win.temp.compile_and_fit(linear, window)

    plot_url = window.plot(linear)

    return plot_url





