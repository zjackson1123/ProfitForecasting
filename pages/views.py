from re import T
from flask import render_template
from Pages import app
from Model import MLmodel

@app.route('/')
@app.route('/home')

def home():
    df = MLmodel.scaledData()
    return render_template(
        "index.html",
        title = "Profit Forecasting",
        plot = df)
        #tables = [df.to_html()],
        #titles = [''])
