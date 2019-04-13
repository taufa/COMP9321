from flask_wtf import FlaskForm
from wtforms import DecimalField, IntegerField, RadioField, SelectField, SubmitField
from wtforms import validators, ValidationError

class PredictionForm(FlaskForm):
	age 		= IntegerField("Age") #, [validators.DataRequired()])
	sex 		= SelectField('Gender', choices=[(1, ""), (1, 'Male'),(0, 'Female')])
	cp			= SelectField('Chest pain type', choices=[(0, ""), (1, 'Typical angina'), (2, 'Atypical angina'), (3, 'Non-anginal pain'), (4, 'Asymptomatic')])
	restbps 	= IntegerField('Resting blood pressure')
	chol		= IntegerField('Serum cholestoral') #, [validators.DataRequired()])
	fbs			= SelectField('Fasting blood sugar', choices=[(0, ""), (1, 'True'), (0, 'False')])
	restecg		= SelectField('Resting electrocardiographic', choices=[(0, ""), (0, 'Normal'), (1, 'having ST-T wave abnormality'), (2, 'Showing probable or definite left ventricular hypertrophy')])
	maxhr		= IntegerField('Heart rate')#, [validators.DataRequired()])
	exang		= SelectField('Exercise induced angina', choices=[(0, ""), (1, 'Yes'), (0, 'No')])
	oldpeak		= IntegerField('ST depression')#, [validators.DataRequired()])
	slope		= SelectField('Slope', choices=[(0, ""), (1, 'Upsloping'), (2, 'Flat'), (3, 'Downsloping')])
	ca			= SelectField('Number of major vessels', choices=[(0, ""), (0, '0'), (1, '1'), (2, '2'), (3, '3')])
	submit		= SubmitField("Predict")
