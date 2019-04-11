README

Design considerations:

- Web app
	- Database
	- Flask
	- Visualisations
	- etc...


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
		The model selected consists of 3 hidden layers and 24 hidden nodes per layer. The model was trained using the following parameters / functions
		- Loss function: Cross Entropy Loss
		- Optimizer: Adam
		- Learning rate:
		- Weight Decay
		- Mini-batch size

	The results (predictions) for each individual algorithm are tallied and the final prediction is the class with the most votes.

