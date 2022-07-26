from flask import render_template
from Pages import app
from Model import MLmodel

@app.route('/')
@app.route('/home')

def home():
    return render_template(
        "index.html",
        title = "Profit Forecasting",
        data = app.send_static_file(MLmodel.scaledData),
        content = "Welcome to the Profit Forecasting Website c: ")
