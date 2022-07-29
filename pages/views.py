from re import T
from flask import render_template
from Pages import app
from Model import MLmodel

@app.route('/')
@app.route('/home')

def home():
    imgs, errors = MLmodel.errorScores()
    return render_template(
        "index.html",
        title = "Profit Forecasting",
        imgs = imgs,
        table = errors)
        #tables = [df.to_html()],
        #titles = [''])
