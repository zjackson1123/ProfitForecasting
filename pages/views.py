from flask import render_template
from Pages import app
from Model import MLmodel

@app.route('/')
@app.route('/home')

def home():
    df, loss, rmse, epochs, batch_size = MLmodel.Predict()
    return render_template(
        "index.html",
        title = "Profit Forecasting",
        plot = df,
        loss = loss, 
        rmse = rmse,
        epochs = epochs,
        batch_size = batch_size)
