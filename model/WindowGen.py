import numpy as np
import matplotlib.pyplot as plot
from io import BytesIO
import base64

class WindowGen():
    def __init__(self, input_width, label_width, shift, 
               train_df, val_df, test_df,
               label_columns=None):
        self.train_df = train_df.iloc[:,0].values
        self.val_df = val_df.iloc[:,0].values
        self.test_df = test_df.iloc[:,0].values
        self.label_columns = label_columns
        if label_columns is not None:
            self.label_columns_indices = {name: i for i, name in enumerate(label_columns)}
        self.column_indices = {name: i for i, name in enumerate(train_df.columns)}
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

    def plot(self, model=None, plot_col = 'Open'):
        inputs, labels = self.example
        plot.figure(figsize=(12,8))
        plot_col_index = self.column_indices[plot_col]
        for n in range(len(inputs)):
            plot.subplot(len(inputs), 1, n+1)
            plot.ylabel=(f'{plot_col} [normed]')
            plot.plot(self.input_indices, inputs[n, :, plot_col_index], label='Inputs', marker='.', zorder=-10)

            if self.label_columns:
                label_col_index = self.label_columns_indices.get(plot_col, None)
            else:
                label_col_index = plot_col_index

            if label_col_index is None:
                continue
            plot.scatter(self.label_indices, labels[n, :, label_col_index], edgecolors='k', labels='Labels', c='#2ca02c', s=64)
            if model is not None:
                predictions = model.input
                plot.scatter(self.label_indices, predictions[n, label_col_index], marker='X', edgecolors='k', labels='Predictions', c='#ff7f0e', s=64)
            if n == 0:
                plot.legend()

        plot.xlabel('Date [d]')
        img = BytesIO()
        plot.saveimg(img, format='png')
        img.seek(0)
        plot_url = base64.b64encode(img.getvalue()).decode('utf-8')
        return plot_url

    def split_window(self, features):
        #1d array needs to be 3d, still a lot to learn
        inputs = features[:, self.input_slice, :]
        labels = features[:, self.labels_slice, :]
        if self.label_columns is not None:
            abels = tf.stack(
            [labels[:, :, self.column_indices[name]] for name in self.label_columns],
            axis=-1)
        inputs.set_shape([None, self.input_width, None])
        labels.set_shape([None, self.label_width, None])

        return inputs, labels



        

