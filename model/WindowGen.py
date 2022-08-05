from pickle import TRUE
import numpy as np
import matplotlib.pyplot as plot
import tensorflow as tf
from tensorflow.keras.constraints import 
import model.PrepData as pd
from sklearn import metrics
from io import BytesIO
import base64

class temp():
    class WindowGen():
        def __init__(self, input_width, label_width, shift, 
                   data,
                   label_columns=None):
            prep = pd.PrepData()
            self.train_data, self.val_data, self.test_data = prep.prep_data(data)
            self.label_columns = label_columns
            if label_columns is not None:
                self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
            self.column_indices = {name: i for i, name in enumerate(self.train_data.columns)}
            self.input_width = input_width
            self.label_width = label_width
            self.shift = shift
            self.total_window_size = input_width + shift 
            self.input_slice = slice(0, input_width)
            self.input_indices = np.arange(self.total_window_size)[self.input_slice]
            self.label_start = self.total_window_size - self.label_width
            self.labels_slice = slice(self.label_start, None)
            self.label_indices = np.arange(self.total_window_size)[self.labels_slice]


        def __repr__(self):
            return '\n' .join([
                   f'Total window size: {self.total_window_size}',
                f'Input indices: {self.input_indices}',
                f'Label indices: {self.label_indices}',
                f'Label column name(s): {self.label_columns}'])          

    def plot(self, model=None, plot_col = 'Open', max_subplots=1):
        inputs, labels = self.example
        revert = pd.PrepData()
        plot.figure(figsize=(12,8))
        plot_col_index = self.column_indices[plot_col]
        max_n = min(max_subplots, len(inputs))
        for n in range(max_n):
            plot.subplot(max_n, 1, n+1)
            plot.ylabel(f'{plot_col} [normed]')
            displayinputs = revert.revert_data(inputs[n,:,:])
            plot.plot(self.input_indices, displayinputs[:, plot_col_index], label='Input', marker='.', zorder=-10)
            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue
            displaylabels = revert.revert_data(labels[n,:,:])
            plot.plot(self.label_indices, displaylabels[:, label_col_index], label='Actual', c='#2ca02c', marker='.', zorder=-10)
            if model is not None:
                predictions = model(inputs)
                predictions = revert.revert_data(predictions[n,:,:])
                plot.plot(self.label_indices, predictions[:, label_col_index], marker='.', label='Prediction', c='#ff7f0e', zorder=-10)
            if n == 0:
                plot.legend()
        img = BytesIO()
        plot.savefig(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
        return plot_url
    WindowGen.plot = plot

    def split_window(self, features):
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            labels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels
    WindowGen.split_window = split_window

    def make_dataset(self, data):
        data = np.array(data, dtype=np.float32)
        global batch_size
        batch_size = 32
        ds = tf.keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=self.total_window_size,
        sequence_stride=1,
        shuffle=True,
        batch_size=batch_size)

        ds = ds.map(self.split_window)

        return ds
    WindowGen.make_dataset = make_dataset

    @property
    def train(self):
        return self.make_dataset(self.train_data)

    @property
    def val(self):
        return self.make_dataset(self.val_data)

    @property
    def test(self):
        return self.make_dataset(self.test_data)

    @property
    def example(self):
        result = getattr(self, '_example', None)
        if result is None:  
            result = next(iter(self.train))  
            self._example = result
        return result
    WindowGen.train = train
    WindowGen.val = val
    WindowGen.test = test
    WindowGen.example = example

    class FeedBack(tf.keras.Model):
        def __init__(self, units, out_steps):
            super().__init__()
            self.out_steps = out_steps
            self.units = units
            self.lstm_cell = tf.keras.layers.LSTMCell(units, activation='tanh', kernel_contraint=tf.keras.constraints.MaxNorm(3))
            self.lstm_rnn = tf.keras.layers.RNN(self.lstm_cell, return_state=True, kernel_contraint=tf.keras.constraints.MaxNorm(3))
            self.dropout = tf.keras.layers.Dropout(0.2)
            self.dense = tf.keras.layers.Dense(1)
            

    def warmup(self, inputs):
      x, *state = self.lstm_rnn(inputs)
      prediction = self.dense(x)
      return prediction, state

    FeedBack.warmup = warmup
    def call(self, inputs, training=None):
      predictions = []
      prediction, state = self.warmup(inputs)
      predictions.append(prediction)
      for n in range(1, self.out_steps):
        x = prediction
        x, state = self.lstm_cell(x, states=state, training=training)
        prediction = self.dense(x)
        predictions.append(prediction)
      predictions = tf.stack(predictions)
      predictions = tf.transpose(predictions, [1, 0, 2])
      return predictions

    FeedBack.call = call

    def compile_and_fit(model, window, patience=7):
        MAX_EPOCHS = 150
        stop = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=patience, mode='min', restore_best_weights=True)
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)
        model.compile(loss=tf.keras.losses.MeanAbsoluteError(),
                optimizer=optimizer,
                metrics=['accuracy'])


        history = model.fit(window.train, epochs=MAX_EPOCHS,
                      validation_data=window.val, 
                      callbacks=stop, verbose=2)
                      
        return history, batch_size

     
