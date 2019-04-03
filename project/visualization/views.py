import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from project import db
from project.models import HeartDisease
from matplotlib import pyplot as plt
from sqlalchemy.orm import sessionmaker
from pandas.plotting import scatter_matrix


def load_data():
    Session = sessionmaker(bind=db.engine)
    session = Session()
    query = session.query(HeartDisease).all()
    data = pd.read_sql(session.query(HeartDisease).statement, session.bind, index_col='id')
    return data

def display():
	data = load_data()
	# print("Shape: ", data.shape)
	# print("Data head(20): ", data.head(20))
	# print("Description: ", data.describe())
	# print("Thal class: ", data.groupby('thal').size())
	data.dropna(how='any', inplace=True)
	df = data[['age', 'maxhr', 'oldpeak', 'chol']]
	df.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
	df.hist()
	scatter_matrix(df)
	plt.show()
