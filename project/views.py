from flask import flash, render_template, request, redirect, url_for, jsonify
from project import app
from .forms import PredictionForm
from .prediction.NeuralNet import predict_thal_nn
from .prediction.Thal_ML_Pred import Predict_using_DecisionTree, Predict_using_kNN, Predict_using_SVC, Predict_using_LogisticRegression
from collections import Counter

@app.route('/', methods=['GET', 'POST'])
def home():
	form = PredictionForm()
	submission_successful = True
	predictions = nn_label = svm_label = knn_label = dt_label = lr_label = ''
	if request.method == 'POST':

		voting = {0: 0, 1: 0, 2: 0}

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
		svm_result = Predict_using_SVC(int(age), int(sex), int(cp), float(restbps), float(chol), int(fbs), int(restecg), float(maxhr), int(exang), float(oldpeak), int(slope), int(ca))
		knn_result = Predict_using_kNN(int(age), int(sex), int(cp), float(restbps), float(chol), int(fbs), int(restecg), float(maxhr), int(exang), float(oldpeak), int(slope), int(ca))
		dt_result = Predict_using_DecisionTree(int(age), int(sex), int(cp), float(restbps), float(chol), int(fbs), int(restecg), float(maxhr), int(exang), float(oldpeak), int(slope), int(ca))
		lr_result = Predict_using_LogisticRegression(int(age), int(sex), int(cp), float(restbps), float(chol), int(fbs), int(restecg), float(maxhr), int(exang), float(oldpeak), int(slope), int(ca))

		voting[int(nn_result)] += 2
		voting[int(dt_result)] += 1.5
		voting[int(knn_result)] += 1.5
		voting[int(svm_result)] += 1
		voting[int(lr_result)] += 1

		print(nn_result, svm_result, knn_result, dt_result, lr_result)
		final_prediction, votes = sorted(voting.items(), key = lambda x: x[1], reverse = True)[0]
		print(final_prediction, votes)

		if 0 in (nn_result, svm_result, knn_result, dt_result, lr_result):
			nn_label = svm_label = knn_label = dt_label = lr_label = 'Normal'
		if 1 in (nn_result, svm_result, knn_result, dt_result, lr_result):
			nn_label = svm_label = knn_label = dt_label = lr_label = 'Fixed Defect'
		if 2 in (nn_result, svm_result, knn_result, dt_result, lr_result):
			nn_label = svm_label = knn_label = dt_label = lr_label = 'Reversable Defect'
	
		predictions = f'Neural Network = {nn_label}, SVM = {svm_label}, k-NN = {knn_label}, Decision Tree = {dt_label}, Logistic Regression = {lr_label}'

		if final_prediction == 0:
			flash("Normal")
		if final_prediction == 1:
			flash("Fixed Defect")
		if final_prediction == 2:
			flash("Reversable Defect")
		return redirect("http://127.0.0.1:5000/#predict")	
	return render_template('home.html', form=form, submission_successful=submission_successful)