from flask import flash, render_template, request, redirect, url_for, jsonify
from project import app
from .forms import PredictionForm
#from .prediction.NeuralNet import predict_thal_nn

@app.route('/', methods=['GET', 'POST'])
def home():
	form = PredictionForm()
	submission_successful = True
	if request.method == 'POST':
		age = request.form.get('age')
		print(age)
		if age == '1':
			flash("Normal")
		if age == '2':
			flash("Fixed Defect")
		if age == '3':
			flash("Reversable Defect")
		return redirect("http://127.0.0.1:5000/#predict")		
	return render_template('home.html', form=form, submission_successful=submission_successful)