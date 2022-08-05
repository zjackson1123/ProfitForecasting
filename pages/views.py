from flask import render_template
from pages import app
from model import MLmodel

@app.route('/')
@app.route('/home')

def home():
    df, t_loss, t_rmse, v_loss, v_rmse, epochs, batch_size = MLmodel.Predict()
    return render_template(
        "index.html",
        title = "Profit Forecasting",
        plot = df,
        t_loss = t_loss, 
        t_rmse = t_rmse,
        v_loss = v_loss,
        v_rmse = v_rmse,
        epochs = epochs,
        batch_size = batch_size)
