from flask import render_template, request, redirect, url_for
from project import app

@app.route('/')
def home():

	return render_template('home.html')

@app.route('/visual/')
def display():
    return "<h1>Visualization</h1>"

@app.route('/predict/')
def predict():
    return "<h1>Prediction</h1>"