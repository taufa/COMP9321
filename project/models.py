import datetime
import pandas as pd
from project import db
from sqlalchemy import event

'''
Source:
https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/heart-disease.names
'''
class HeartDisease(db.Model):
    __tablename__ = "heart_disease"

    id          = db.Column(db.Integer, primary_key=True) # primary_key
    age         = db.Column(db.Float) # age in years
    sex         = db.Column(db.Float) # sex(1 = male; 0 = female)
    cp          = db.Column(db.Float) # chest pain type(1:typical angina, 2:atypical angina, 3:non-anginal pain, 4:asymptomatic)
    rest_bp     = db.Column(db.Float) # resting blood pressure
    chol        = db.Column(db.Float) # serum cholestoral in mg/dl
    fbs         = db.Column(db.Float) # fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
    rest_ec     = db.Column(db.Float) # resting electrocardiographic results(0: normal, 1: having ST-T wave abnormality (T wave inversions and/or ST elevation or depression of > 0.05 mV), 2: showing probable or definite left ventricular hypertrophy by Estes' criteria)
    max_hr      = db.Column(db.Float) # maximum heart rate achieved
    exang       = db.Column(db.Float) # exercise induced angina (1 = yes; 0 = no)
    oldpeak     = db.Column(db.Float) # ST depression induced by exercise relative to rest
    slope       = db.Column(db.Float) # slope: the slope of the peak exercise ST segment(1: upsloping, 2: flat, 3: downsloping)
    ca          = db.Column(db.Float) # number of major vessels (0-3) colored by flourosopy
    thal        = db.Column(db.Float) # 3 = normal; 6 = fixed defect; 7 = reversable defect
    num         = db.Column(db.Float) # have heart disease(>=1 = yes; 0 = no)

    def __repr__(self):
        return self.id

def load_data(db_file):
    data = pd.read_csv(db_file, header=None)
    data.columns = ['age', 'sex', 'cp', 'rest_bp', 'chol', 'fbs', 'rest_ec', 'max_hr', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'num']
    data.to_sql(HeartDisease.__table__.name, con=db.engine, if_exists='replace', index=False)