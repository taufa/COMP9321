from flask import render_template, request, redirect, url_for
from project import app

@app.route('/')
def home():
	return "<h1>COMP9321</h1>"

@app.route('/display/')
def display():
    return "<h1>Visualization</h1>"

@app.route('/predict/')
def predict():
    return "<h1>Prediction</h1>"