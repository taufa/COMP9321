import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from flask_bootstrap import Bootstrap
from flask_wtf.csrf import CSRFProtect, CSRFError

app = Flask(__name__)
Bootstrap(app)
csrf = CSRFProtect(app)

db_path = os.path.join(os.path.dirname(__file__), 'data.db')
db_uri = 'sqlite:///{}'.format(db_path)

app.config.update(dict(
	SQLALCHEMY_DATABASE_URI=db_uri,
	SQLALCHEMY_TRACK_MODIFICATIONS=True,
	SECRET_KEY="ynots-secret-key",
	WTF_CSRF_SECRET_KEY="ynots-csrf-secret-key",
	WTF_CSRF_TIME_LIMIT=3600,
	WTF_CSRF_ENABLED = False,
	))

db = SQLAlchemy(app)
migrate = Migrate(app, db)

from .models import HeartDisease, init_db
from .prediction.FeatureImportance import feature_chi2
from .visualization.views import display
#init_db()
#feature_chi2()
display()

from .views import *
