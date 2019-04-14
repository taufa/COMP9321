README

Design considerations:

- Web app
	- Database
		SQLite was chosen as our database with SQLAlchemy as the Python SQL toolkit and ORM. The data file is initially loaded into our database and retrieved from the database as an ORM query for each specific task and models.
	- Flask
		Used flask 1.0.2 including flask-forms for our predition form, flask-sqlalchemy for back-end and flask-bootstrap for frontend.
	- Visualisations
		Plots and graphs were done using pandas plot and matplotlib libraries
		- labelled features used stacked bar charts and a normal bar graph to show correlations between age and gender with all the other features.
		- continuous values included a scatter plot and a bar graph with a error bar to show the standard deviation of the feature values against age and gender.

- Feature Importance
	The features for this dataset were ranked in order of importance using the chi^2 algorithm.

- Multi-class classification 
	Our designed classifier combines several base algorithms to produce one predictive model (ensemble). The algorithms used in our design are:

	- Decission Tree:
		Model description

	- Support Vector Machine:
		Model description

	- K Nearest Neighbour:
		Model description

	- Logistic Regression:
		Model description

	- Neural Network:
		The model selected consists of 4 hidden layers and 6 hidden nodes per layer. The model was trained using the following parameters / functions
		- Loss function: Cross Entropy Loss
		- Optimizer: Adam
		- Learning rate: 0.01
		- Weight Decay: 0
		- Mini-batch size: 16
		- Training epochs: 78

	The results (predictions) for each individual algorithm are tallied and the final prediction is the class with the most votes.

