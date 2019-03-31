import datetime
import pandas as pd
from project import db
from sqlalchemy import event
import numpy as np

'''
Source:
https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names
'''
class HeartDisease(db.Model):
    __tablename__ = "heart_disease"

    id          = db.Column(db.Integer, primary_key=True) # primary_key
    age         = db.Column(db.Float, nullable=True) # age in years
    sex         = db.Column(db.Float, nullable=True) # sex(1 = male; 0 = female)
    cp          = db.Column(db.Float, nullable=True) # chest pain type(1:typical angina, 2:atypical angina, 3:non-anginal pain, 4:asymptomatic)
    rest_bp     = db.Column(db.Float, nullable=True) # resting blood pressure
    chol        = db.Column(db.Float, nullable=True) # serum cholestoral in mg/dl
    fbs         = db.Column(db.Float, nullable=True) # fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    rest_ec     = db.Column(db.Float, nullable=True) # resting electrocardiographic results(0: normal, 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), 2: showing probable or definite left ventricular hypertrophy by Estes' criteria)
    max_hr      = db.Column(db.Float, nullable=True) # maximum heart rate achieved
    exang       = db.Column(db.Float, nullable=True) # exercise induced angina (1 = yes; 0 = no)
    oldpeak     = db.Column(db.Float, nullable=True) # ST depression induced by exercise relative to rest
    slope       = db.Column(db.Float, nullable=True) # slope: the slope of the peak exercise ST segment(1: upsloping, 2: flat, 3: downsloping)
    ca          = db.Column(db.Float, nullable=True) # number of major vessels (0-3) colored by flourosopy
    thal        = db.Column(db.Float, nullable=True) # 3 = normal; 6 = fixed defect; 7 = reversable defect
    num         = db.Column(db.Float, nullable=True) # have heart disease(>=1 = yes; 0 = no)

    def __init__(self, age=None, sex=None, cp=None, rest_bp=None, chol=None, fbs=None, rest_ec=None, max_hr=None, exang=None, oldpeak=None, slope=None, ca=None, thal=None, num=None):
        self.data = (age, sex, cp, rest_bp, chol, fbs, rest_ec, max_hr, exang, oldpeak, slope, ca, thal, num)

    def __repr__(self):
        return (self.age, self.sex, self.cp, self.rest_bp, self.chol, self.fbs, self.rest_ec, self.max_hr, self.exang, self.oldpeak, self.slope, self.ca, self.thal, self.num)

'''
Load the data file into sqlite db.
:param db_file: directory of data file
:return: dataframe with ? values as NaN
'''
def load_data(db_file='data/processed.cleveland.data'):
    data = pd.read_csv(db_file, header=None, float_precision='high')
    data.replace(['?'], [None], inplace=True)
    data.columns = ['age', 'sex', 'cp', 'rest_bp', 'chol', 'fbs', 'rest_ec', 'max_hr', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    data.to_sql(HeartDisease.__table__.name, con=db.engine, if_exists='replace', index=True, index_label='id')
    data.replace(['?'], np.nan, inplace=True)
    return data