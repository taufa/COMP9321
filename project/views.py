from flask import flash, render_template, request, redirect, url_for, jsonify
from project import app
from .forms import PredictionForm
from .prediction.NeuralNet import predict_thal_nn
from .prediction.Thal_ML_Pred import Predict_using_DecisionTree, Predict_using_kNN, Predict_using_SVC, Predict_using_LogisticRegression


@app.route('/', methods=['GET', 'POST'])
def home():
	form = PredictionForm()
	submission_successful = True

	if request.method == 'POST':
		age		= request.form.get('age')
		sex		= request.form.get('sex')
		cp		= request.form.get('cp')
		restbps	= request.form.get('restbps')
		chol	= request.form.get('chol')
		fbs		= request.form.get('fbs')
		restecg	= request.form.get('restecg')
		maxhr	= request.form.get('maxhr')
		exang	= request.form.get('exang')
		oldpeak	= request.form.get('oldpeak')
		slope	= request.form.get('slope')
		ca		= request.form.get('ca')

		#print(age, sex, cp, restbps, chol, fbs, restecg, maxhr, exang, oldpeak, slope, ca)
		#print(result)

		nn_result = predict_thal_nn(int(age), int(sex), int(cp), float(restbps), float(chol), int(fbs), int(restecg), float(maxhr), int(exang), float(oldpeak), int(slope), int(ca))
		svc_result = Predict_using_SVC(int(age), int(sex), int(cp), float(restbps), float(chol), int(fbs), int(restecg), float(maxhr), int(exang), float(oldpeak), int(slope), int(ca))
		knn_result = Predict_using_kNN(int(age), int(sex), int(cp), float(restbps), float(chol), int(fbs), int(restecg), float(maxhr), int(exang), float(oldpeak), int(slope), int(ca))
		dt_result = Predict_using_DecisionTree(int(age), int(sex), int(cp), float(restbps), float(chol), int(fbs), int(restecg), float(maxhr), int(exang), float(oldpeak), int(slope), int(ca))
		lr_result = Predict_using_LogisticRegression(int(age), int(sex), int(cp), float(restbps), float(chol), int(fbs), int(restecg), float(maxhr), int(exang), float(oldpeak), int(slope), int(ca))

		print(nn_result, svc_result, knn_result, dt_result, lr_result)
		
		if nn_result == 0 or nn_result == 3:
			flash("Normal")
		if nn_result == 1 or nn_result == 6:
			flash("Fixed Defect")
		if nn_result == 2 or nn_result == 7:
			flash("Reversable Defect")

		return redirect("http://127.0.0.1:5000/#predict")		
	return render_template('home.html', form=form, submission_successful=submission_successful)