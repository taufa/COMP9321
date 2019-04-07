from flask import flash, render_template, request, redirect, url_for, jsonify
from project import app
from .forms import PredictionForm

@app.route('/', methods=['GET', 'POST'])
def home():
	form = PredictionForm()
   
	if request.method == 'POST':
		if not form.validate_on_submit():
			flash('Some fields are required.')
		else:
			data = request.form['input_name']
			return render_template('home.html', form=form)
	return render_template('home.html', form=form)

@app.route('/visual/')
def display():
    return "<h1>Visualization</h1>"

@app.route('/predict/')
def predict():
    return "<h1>Prediction</h1>"