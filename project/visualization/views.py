import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import os
import pandas as pd
import plotly.figure_factory as ff
import plotly.plotly as py
from matplotlib import pylab
from matplotlib import pyplot as plt
from pandas.plotting import scatter_matrix
from project import db
from project.models import HeartDisease
from sqlalchemy.orm import sessionmaker


def load_data():
	Session = sessionmaker(bind=db.engine)
	session = Session()
	query = session.query(HeartDisease).all()
	data = pd.read_sql(session.query(HeartDisease).statement, session.bind, index_col='id')
	data.dropna(how='any', inplace=True)
	data.drop('num', axis=1, inplace=True)
	data = data.apply(pd.to_numeric)
	return label_data(data)

def label_data(data):
	sex_dict = {0: "female", 1: "male"}
	cp_dict = {1: "typical angina", 2: "atypical angina", 3: "non-anginal pain", 4: "asymptomatic"}
	fbs_dict = {0: "false", 1: "true"}
	restecg_dict = {0: "normal", 1: "ST-T wave abnormality", 2: "showing probable"}
	exang_dict = {1: "yes", 0: "no"}
	slope_dict = {1: "upsloping", 2: "flat", 3: "downsloping"}
	thal_dict = {3: "normal", 6: "fixed defect", 7: "reversable defect"}
	data["sex"].replace(sex_dict, inplace=True)
	data["cp"].replace(cp_dict, inplace=True)
	data["fbs"].replace(fbs_dict, inplace=True)
	data["restecg"].replace(restecg_dict, inplace=True)
	data["exang"].replace(exang_dict, inplace=True)
	data["slope"].replace(slope_dict, inplace=True)
	data["thal"].replace(thal_dict, inplace=True)
	data.rename(index=str, columns={
		"sex": "gender", 
		"cp": "chest_pain_type", 
		"restbps": "resting_blood_pressure",
		"chol": "serum_cholestoral",
		"fbs": "fasting_blood_sugar",
		"restecg": "resting_electrocardiographic_results",
		"maxhr": "maximum_heart_rate_achieved",
		"exang": "exercise_induced_angina",
		"slope": "slope_peak_exercise_ST_segment",
		"ca": "number_of_major_vessels"}, inplace=True)
	return data

def display():
	data = load_data()
	# # print("Shape: ", data.shape)
	# # print("Data head(20): ", data.head(20))
	# # print("Description: ", data.describe())
	# # print("Thal class: ", data.groupby('thal').size())

	# thalassemia on different gender
	stacked_bar_gender(data)
	stacked_bar_thal_gender_norm(data)

	# thalassemia over age
	line_plot_age_gender(data)
	hist_age_thal(data)

	# check pain 
	stacked_bar_cp_gender(data)
	line_plot_cp_age(data)
	bar_graph_cp(data)
	hist_age_cp(data)

	# fasting blood sugar
	stacked_bar_fbs_gender(data)
	line_plot_fbs_age(data)
	bar_graph_fbs(data)
	hist_age_fbs(data)

	# resting electrocardiographic results
	stacked_bar_restecg_gender(data)
	line_plot_restecg_age(data)
	bar_graph_restecg(data)
	hist_age_restecg(data)

	# exercise induced angina
	stacked_bar_exang_gender(data)
	line_plot_exang_age(data)
	bar_graph_exang(data)
	hist_age_exang(data)

	# exercise induced angina
	stacked_bar_slope_gender(data)
	line_plot_slope_age(data)
	bar_graph_slope(data)
	hist_age_slope(data)

	# # scatter matrix on continuous feature values
	# features_scatter_matrix(data)

	# serum cholestoral
	scatter_age_chol(data)
	chol_gender_thal_plot(data)

	# maximum heart rate achieved
	scatter_age_maxhr(data)
	maxhr_gender_thal_plot(data)

	# resting blood pressure
	scatter_age_restbp(data)
	restbp_gender_thal_plot(data)

	# ST depression induced by exercise relative to rest
	scatter_age_oldpeak(data)
	oldpeak_gender_thal_plot(data)

	# number of major vessels (0-3) colored by flourosopy
	scatter_age_ca(data)
	ca_gender_thal_plot(data)
'''
save figure to static folder
'''
def save_figure(filename):
	dir_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'static'))
	plt.savefig(f'{dir_path}\\img\\{filename}.png', bbox_inches='tight')
	plt.clf()

'''
Age against gender
'''
def hist_age_thal(data):
	df = data[['age', 'thal']]
	df = data.groupby([pd.cut(df.age, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]), 'thal'])['age'].count().unstack('thal').fillna(0)
	sb = df[['fixed defect', 'normal', 'reversable defect']].plot(kind='bar', stacked=True, rot=0, title="Thalassemia type count over Age")
	sb.set_xlabel("Age")
	sb.set_ylabel("Thalassemia Count")
	save_figure('hist_age_thal')
	#plt.show()

def line_plot_age_gender(data):
	df = data[['age', 'thal']]
	df.groupby('thal').age.plot(
		kind='kde', legend=True, title="Thalassemia density by Age")
	save_figure('line_plot_age_gender')
	#plt.show()

def stacked_bar_gender(data):
	df = data.groupby(['thal', 'gender'])['thal'].count().unstack('gender').fillna(0)
	sb = df[['female','male']].plot(kind='bar', stacked=True, rot=0, title="Gender count by Thalassemia type")
	sb.set_xlabel("Thalassemia")
	sb.set_ylabel("Count")
	save_figure('stacked_bar_gender')
	#plt.show()

def stacked_bar_thal_gender_norm(data):
	sb = data.groupby(['gender','thal']).size().groupby(level=0).apply(
		lambda x: 100 * x / x.sum()).unstack().plot(kind='bar',stacked=True, rot=0, title="Gender on Thalassemia types (normalized)")
	sb.set_xlabel("Gender")
	sb.set_ylabel("Percentage")
	plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
	save_figure('stacked_bar_thal_gender_norm')
	#plt.show()

'''
chest pain types
'''
def stacked_bar_cp_gender(data):
	sbn = data.groupby(['gender','chest_pain_type']).size().groupby(level=0).apply(
		lambda x: 100 * x / x.sum()).unstack().plot(kind='bar',stacked=True, rot=0, title="Chest pain type on Gender (Normalized)")
	sbn.set_xlabel("Gender")
	sbn.set_ylabel("Percentage")
	plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
	save_figure('stacked_bar_cp_gender')
	#plt.show()

def hist_age_cp(data):
	df = data[['age', 'chest_pain_type']]
	df = data.groupby([pd.cut(df.age, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]), 'chest_pain_type'])['age'].count().unstack('chest_pain_type').fillna(0)
	sb = df[['typical angina', 'atypical angina', 'non-anginal pain', 'asymptomatic']].plot(kind='bar', stacked=True, rot=0, title="Chest pain over Age")
	sb.set_xlabel("Age")
	sb.set_ylabel("Chest pain count")
	save_figure('hist_age_cp')

def line_plot_cp_age(data):
	df = data[['age', 'chest_pain_type']]
	df.groupby('chest_pain_type').age.plot(
		x='Age', kind='kde', title="Chest pain density over Age")
	save_figure('line_plot_cp_age')
	#plt.show()

def bar_graph_cp(data):
	df = data.groupby(['thal', 'chest_pain_type'])['thal'].count().unstack('chest_pain_type').fillna(0)
	sb = df[['asymptomatic','atypical angina', 'non-anginal pain', 'typical angina']].plot(kind='bar', 
		rot=0, title="Thalassemia on Chest Pain")
	sb.set_xlabel("Chest pain type")
	sb.set_ylabel("Count")
	save_figure('bar_graph_cp')
	#plt.show()

'''
maximum heart rate
'''
def scatter_age_maxhr(data):
	df = data[['age', 'maximum_heart_rate_achieved']]
	sp = df.plot.scatter(x='age', y='maximum_heart_rate_achieved');
	sp.set_title("Heart rate over Age")
	save_figure('scatter_age_maxhr')
	#plt.show()

def maxhr_gender_thal_plot(data):
	df = data[['thal', 'gender', 'maximum_heart_rate_achieved']]
	df['female'] = df.loc[df['gender'] == 'female', 'maximum_heart_rate_achieved']
	df['male'] = df.loc[df['gender'] == 'male', 'maximum_heart_rate_achieved']

	means = df.groupby(['thal']).mean().fillna(0)
	errors = df.groupby(['thal']).std().fillna(0)
	means.drop(['maximum_heart_rate_achieved'], axis=1, inplace=True)
	errors.drop(['maximum_heart_rate_achieved'], axis=1, inplace=True)
	_, ax = plt.subplots()
	plot = means.plot.bar(yerr=errors, ax=ax, rot=0, title="Heart rate for Gender and Thalassemia")
	plot.set_xlabel("Thalassemia")
	plot.set_ylabel("Heart rate")
	save_figure('maxhr_gender_thal_plot')

'''
fasting blood sugar
'''
def stacked_bar_fbs_gender(data):
	sbn = data.groupby(['gender','fasting_blood_sugar']).size().groupby(level=0).apply(
		lambda x: 100 * x / x.sum()).unstack().plot(kind='bar',stacked=True, rot=0, title="Fasting blood sugar type on Gender (Normalized)")
	sbn.set_xlabel("Gender")
	sbn.set_ylabel("Percentage")
	plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
	save_figure('stacked_bar_fbs_gender')
	#plt.show()

def hist_age_fbs(data):
	df = data[['age', 'fasting_blood_sugar']]
	df = data.groupby([pd.cut(df.age, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]), 'fasting_blood_sugar'])['age'].count().unstack('fasting_blood_sugar').fillna(0)
	sb = df[['false', 'true']].plot(kind='bar', stacked=True, rot=0, title="Fasting blood sugar over Age")
	sb.set_xlabel("Age")
	sb.set_ylabel("Fasting blood sugar count")
	save_figure('hist_age_fbs')

def line_plot_fbs_age(data):
	df = data[['age', 'fasting_blood_sugar']]
	df.groupby('fasting_blood_sugar').age.plot(
		x='Age', kind='kde', title="Fasting blood sugar density over Age")
	save_figure('line_plot_fbs_age')
	#plt.show()

def bar_graph_fbs(data):
	df = data.groupby(['thal', 'fasting_blood_sugar'])['thal'].count().unstack('fasting_blood_sugar').fillna(0)
	sb = df[['true','false']].plot(kind='bar', rot=0, title="Thalassemia on fasting blood sugar")
	sb.set_xlabel("Fasting blood sugar type")
	sb.set_ylabel("Count")
	save_figure('bar_graph_fbs')
	#plt.show()

'''
resting blood pressure
'''
def scatter_age_restbp(data):
	df = data[['age', 'resting_blood_pressure']]
	sp = df.plot.scatter(x='age', y='resting_blood_pressure');
	sp.set_title("Resting blood pressure over Age")
	save_figure('scatter_age_restbp')
	#plt.show()

def restbp_gender_thal_plot(data):
	df = data[['thal', 'gender', 'resting_blood_pressure']]
	df['female'] = df.loc[df['gender'] == 'female', 'resting_blood_pressure']
	df['male'] = df.loc[df['gender'] == 'male', 'resting_blood_pressure']

	means = df.groupby(['thal']).mean().fillna(0)
	errors = df.groupby(['thal']).std().fillna(0)
	means.drop(['resting_blood_pressure'], axis=1, inplace=True)
	errors.drop(['resting_blood_pressure'], axis=1, inplace=True)
	_, ax = plt.subplots()
	plot = means.plot.bar(yerr=errors, ax=ax, rot=0, title="Resting blood pressure for Gender and Thalassemia")
	plot.set_xlabel("Thalassemia")
	plot.set_ylabel("Resting blood pressure")
	save_figure('restbp_gender_thal_plot')

'''
resting electrocardiographic results
'''
def stacked_bar_restecg_gender(data):
	sbn = data.groupby(['gender','resting_electrocardiographic_results']).size().groupby(level=0).apply(
		lambda x: 100 * x / x.sum()).unstack().plot(kind='bar',stacked=True, rot=0, title="Resting electrocardiographic on Gender (Normalized)")
	sbn.set_xlabel("Gender")
	sbn.set_ylabel("Percentage")
	plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
	save_figure('stacked_bar_restecg_gender')
	#plt.show()

def hist_age_restecg(data):
	df = data[['age', 'resting_electrocardiographic_results']]
	df = data.groupby([pd.cut(df.age, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]), 'resting_electrocardiographic_results'])['age'].count().unstack('resting_electrocardiographic_results').fillna(0)
	sb = df[['normal', 'ST-T wave abnormality', 'showing probable']].plot(kind='bar', stacked=True, rot=0, title="Resting electrocardiographic over Age")
	sb.set_xlabel("Age")
	sb.set_ylabel("Resting electrocardiographic sugar count")
	save_figure('hist_age_restecg')

def line_plot_restecg_age(data):
	df = data[['age', 'resting_electrocardiographic_results']]
	df.groupby('resting_electrocardiographic_results').age.plot(
		x='Age', kind='kde', title="Resting electrocardiographic density over Age")
	save_figure('line_plot_restecg_age')
	#plt.show()

def bar_graph_restecg(data):
	df = data.groupby(['thal', 'resting_electrocardiographic_results'])['thal'].count().unstack('resting_electrocardiographic_results').fillna(0)
	sb = df[['normal','ST-T wave abnormality', 'showing probable']].plot(kind='bar', rot=0, title="Thalassemia on resting electrocardiographic")
	sb.set_xlabel("Resting electrocardiographic results")
	sb.set_ylabel("Count")
	save_figure('bar_graph_restecg')
	#plt.show()


'''
serum cholestoral
'''
def scatter_age_chol(data):
	df = data[['age', 'serum_cholestoral']]
	sp = df.plot.scatter(x='age', y='serum_cholestoral');
	sp.set_title("Serum cholestoral over Age")
	save_figure('scatter_age_chol')
	#plt.show()

def chol_gender_thal_plot(data):
	df = data[['thal', 'gender', 'serum_cholestoral']]
	df['female'] = df.loc[df['gender'] == 'female', 'serum_cholestoral']
	df['male'] = df.loc[df['gender'] == 'male', 'serum_cholestoral']

	means = df.groupby(['thal']).mean().fillna(0)
	errors = df.groupby(['thal']).std().fillna(0)
	means.drop(['serum_cholestoral'], axis=1, inplace=True)
	errors.drop(['serum_cholestoral'], axis=1, inplace=True)
	_, ax = plt.subplots()
	plot = means.plot.bar(yerr=errors, ax=ax, rot=0, title="Serum cholestoral for Gender and Thalassemia")
	plot.set_xlabel("Thalassemia")
	plot.set_ylabel("Serum cholestoral")
	save_figure('chol_gender_thal_plot')


'''
exercise induced angina
'''
def stacked_bar_exang_gender(data):
	sbn = data.groupby(['gender','exercise_induced_angina']).size().groupby(level=0).apply(
		lambda x: 100 * x / x.sum()).unstack().plot(kind='bar',stacked=True, rot=0, title="Exercise induced angina on Gender (Normalized)")
	sbn.set_xlabel("Gender")
	sbn.set_ylabel("Percentage")
	plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
	save_figure('stacked_bar_exang_gender')
	#plt.show()

def hist_age_exang(data):
	df = data[['age', 'exercise_induced_angina']]
	df = data.groupby([pd.cut(df.age, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]), 'exercise_induced_angina'])['age'].count().unstack('exercise_induced_angina').fillna(0)
	sb = df[['yes', 'no']].plot(kind='bar', stacked=True, rot=0, title="Exercise induced angina over Age")
	sb.set_xlabel("Age")
	sb.set_ylabel("Exercise induced angina count")
	save_figure('hist_age_exang')

def line_plot_exang_age(data):
	df = data[['age', 'exercise_induced_angina']]
	df.groupby('exercise_induced_angina').age.plot(
		x='Age', kind='kde', title="Exercise induced angina density over Age")
	save_figure('line_plot_exang_age')
	#plt.show()

def bar_graph_exang(data):
	df = data.groupby(['thal', 'exercise_induced_angina'])['thal'].count().unstack('exercise_induced_angina').fillna(0)
	sb = df[['yes', 'no']].plot(kind='bar', rot=0, title="Thalassemia on exercise induced angina")
	sb.set_xlabel("Exercise induced angina")
	sb.set_ylabel("Count")
	save_figure('bar_graph_exang')
	#plt.show()

'''
ST depression induced by exercise
'''
def scatter_age_oldpeak(data):
	df = data[['age', 'oldpeak']]
	sp = df.plot.scatter(x='age', y='oldpeak');
	sp.set_title("ST depression over Age")
	save_figure('scatter_age_oldpeak')
	#plt.show()

def oldpeak_gender_thal_plot(data):
	df = data[['thal', 'gender', 'oldpeak']]
	df['female'] = df.loc[df['gender'] == 'female', 'oldpeak']
	df['male'] = df.loc[df['gender'] == 'male', 'oldpeak']

	means = df.groupby(['thal']).mean().fillna(0)
	errors = df.groupby(['thal']).std().fillna(0)
	means.drop(['oldpeak'], axis=1, inplace=True)
	errors.drop(['oldpeak'], axis=1, inplace=True)
	_, ax = plt.subplots()
	plot = means.plot.bar(yerr=errors, ax=ax, rot=0, title="ST depression for Gender and Thalassemia")
	plot.set_xlabel("Thalassemia")
	plot.set_ylabel("ST depression")
	save_figure('oldpeak_gender_thal_plot')

'''
slope
'''
def stacked_bar_slope_gender(data):
	sbn = data.groupby(['gender','slope_peak_exercise_ST_segment']).size().groupby(level=0).apply(
		lambda x: 100 * x / x.sum()).unstack().plot(kind='bar',stacked=True, rot=0, title="The slope of the peak exercise ST segment on Gender (Normalized)")
	sbn.set_xlabel("Gender")
	sbn.set_ylabel("Percentage")
	plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
	save_figure('stacked_bar_slope_gender')
	#plt.show()

def hist_age_slope(data):
	df = data[['age', 'slope_peak_exercise_ST_segment']]
	df = data.groupby([pd.cut(df.age, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]), 'slope_peak_exercise_ST_segment'])['age'].count().unstack('slope_peak_exercise_ST_segment').fillna(0)
	sb = df[['upsloping', 'flat', 'downsloping']].plot(kind='bar', stacked=True, rot=0, title="Slope peak exercise over Age")
	sb.set_xlabel("Age")
	sb.set_ylabel("Slope peak exercise count")
	save_figure('hist_age_slope')

def line_plot_slope_age(data):
	df = data[['age', 'slope_peak_exercise_ST_segment']]
	df.groupby('slope_peak_exercise_ST_segment').age.plot(
		x='Age', kind='kde', title="The slope of the peak exercise ST segment density over Age")
	save_figure('line_plot_slope_age')
	#plt.show()

def bar_graph_slope(data):
	df = data.groupby(['thal', 'slope_peak_exercise_ST_segment'])['thal'].count().unstack('slope_peak_exercise_ST_segment').fillna(0)
	sb = df[['upsloping', 'flat', 'downsloping']].plot(kind='bar', rot=0, title="Thalassemia on the slope of the peak exercise ST segment")
	sb.set_xlabel("The slope of the peak exercise ST segment")
	sb.set_ylabel("Count")
	save_figure('bar_graph_slope')
	#plt.show()

'''
number of major vessels (0-3) colored by flourosopy
'''
def scatter_age_ca(data):
	df = data[['age', 'number_of_major_vessels']]
	sp = df.plot.scatter(x='age', y='number_of_major_vessels');
	sp.set_title("number of major vessels over Age")
	save_figure('scatter_age_ca')
	#plt.show()

def ca_gender_thal_plot(data):
	df = data[['thal', 'gender', 'number_of_major_vessels']]
	df['female'] = df.loc[df['gender'] == 'female', 'number_of_major_vessels']
	df['male'] = df.loc[df['gender'] == 'male', 'number_of_major_vessels']

	means = df.groupby(['thal']).mean().fillna(0)
	errors = df.groupby(['thal']).std().fillna(0)
	means.drop(['number_of_major_vessels'], axis=1, inplace=True)
	errors.drop(['number_of_major_vessels'], axis=1, inplace=True)
	_, ax = plt.subplots()
	plot = means.plot.bar(yerr=errors, ax=ax, rot=0, title="Number of major vessels for Gender and Thalassemia")
	plot.set_xlabel("Thalassemia")
	plot.set_ylabel("Number of major vessels")
	save_figure('ca_gender_thal_plot')

'''
features scatter matrix
'''
def features_scatter_matrix(data):
	label = ['age', 
		'gender', 
		'chest_pain_type', 
		'resting_blood_pressure', 
		'serum_cholestoral', 
		'fasting_blood_sugar', 
		'resting_electrocardiographic_results', 
		'maximum_heart_rate_achieved', 
		'exercise_induced_angina', 
		'oldpeak', 
		'slope_peak_exercise_ST_segment',
		'number_of_major_vessels', 
		'thal']
	df = data[label]
	sm = scatter_matrix(df, alpha=0.5, figsize=(20, 10), diagonal='kde')

	#y labels
	[plt.setp(item.yaxis.get_label(), 'size', 9) for item in sm.ravel()]

	#x labels
	[plt.setp(item.xaxis.get_label(), 'size', 9) for item in sm.ravel()]

	#Change label rotation
	#[s.xaxis.label.set_rotation(0) for s in sm.reshape(-1)]
	[s.yaxis.label.set_rotation(75) for s in sm.reshape(-1)]

	# may need to offset label when rotating to prevent overlap of figure
	[s.get_yaxis().set_label_coords(-0.3,0.5) for s in sm.reshape(-1)]

	# may need to offset label when rotating to prevent overlap of figure
	[s.get_yaxis().set_label_coords(-0.3,0.5) for s in sm.reshape(-1)]
	save_figure('features_scatter_matrix')

