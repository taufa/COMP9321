import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate

app = Flask(__name__)

db_path = os.path.join(os.path.dirname(__file__), 'data.db')
db_uri = 'sqlite:///{}'.format(db_path)

app.config.update(dict(
	SQLALCHEMY_DATABASE_URI=db_uri,
	SQLALCHEMY_TRACK_MODIFICATIONS=True,
	TESTING=True,
	))

db = SQLAlchemy(app)
migrate = Migrate(app, db)

from .models import HeartDisease, load_data
from .prediction.FeatureImportance import feature_chi2
db.create_all()
load_data()
feature_chi2()

from .views import *
