import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import random_split, DataLoader, TensorDataset
from torch.autograd import Variable
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import os
from sqlalchemy.orm import sessionmaker
from project import db
from project.models import HeartDisease


class NeuralNet_1H(nn.Module):
    '''Custom Neural Network with 1 Hidden layer'''
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super(NeuralNet_1H, self).__init__()
        self.fc1 = nn.Linear(input_nodes, hidden_nodes)     # 12 features into 24 nodes in the hidden layer
        self.fc2 = nn.Linear(hidden_nodes, output_nodes)      # 24 nodes in the hidden layer to 3 output nodes

    def forward(self, x):
        x = torch.tanh(self.fc1(x))          # Using hyperbolic tangent as activation function
        output = F.log_softmax(self.fc2(x), dim=1)
        return output

class NeuralNet_2H(nn.Module):
    '''Custom Neural Network with 2 Hidden layers'''
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super(NeuralNet_2H, self).__init__()
        self.fc1 = nn.Linear(input_nodes, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc3 = nn.Linear(hidden_nodes, output_nodes)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))          # Using hyperbolic tangent as activation function
        x = torch.tanh(self.fc2(x))
        output = F.log_softmax(self.fc3(x), dim=1)
        return output

class NeuralNet_3H(nn.Module):
    '''Custom Neural Network with 3 Hidden layers'''
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super(NeuralNet_3H, self).__init__()
        self.fc1 = nn.Linear(input_nodes, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc3 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc4 = nn.Linear(hidden_nodes, output_nodes)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))          # Using hyperbolic tangent as activation function
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        output = F.log_softmax(self.fc4(x), dim=1)
        return output

class NeuralNet_4H(nn.Module):
    '''Custom Neural Network with 4 Hidden layers'''
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super(NeuralNet_4H, self).__init__()
        self.fc1 = nn.Linear(input_nodes, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc3 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc4 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc5 = nn.Linear(hidden_nodes, output_nodes)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))          # Using hyperbolic tangent as activation function
        x = torch.tanh(self.fc2(x))
        x = torch.tanh(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        output = F.log_softmax(self.fc5(x), dim=1)
        return output

def load_data():
    '''
    Load data from database and returns it as a dataframe
    '''
    Session = sessionmaker(bind=db.engine)
    session = Session()
    query = session.query(HeartDisease).all()
    data = pd.read_sql(session.query(HeartDisease).statement, session.bind, index_col='id')
    session.close()
    return data

def clean_data(data):
    '''
    Cleans the data by eliminating the NA and converting the values on the 'thal' column to values in
    {0, 1, 2}
    :param data: Dataframe containing the raw data
    :return: Dataframe containing the clean data (without NA)
    '''
    # data.replace('?', np.nan, inplace=True)
    data.dropna(how='any', inplace=True)
    data.iloc[:, 12] = data.iloc[:, 12].astype(np.float64)   # converting the 'thal' column to float64 type
    data.iloc[:, 11] = data.iloc[:, 11].astype(np.float64)  # converting the 'ca' column to float64 type
    mapping_dict = {3: 0, 6: 1, 7: 2}  # this will convert the thal values to 0, 1 or 2
    data.iloc[:, 12] = data.iloc[:, 12].apply(lambda x: mapping_dict[x])
    return data

def standardise_data(data):
    '''
    Standardises the data using a Z-normalisation
    :param data: Dataframe containing the data
    :return: features_scaled: Dataframe containing the standardised features
             target: Dataframe containing the 'thal' column (target)
             mu: Series containing the means for every feature
             sigma: Series containing the standard deviations for every feature
    '''
    data = data.astype(np.float64)  # converting the entire dataframe to float64 data type
    scaler = StandardScaler()
    feature_names = data.columns[:-2]
    target = data.iloc[:, 12]
    features = data.iloc[:,:-2]   # the features are the rest of the columns
    mu = features.mean()
    sigma = features.std()
    features_scaled = scaler.fit_transform(features)
    features_scaled = pd.DataFrame(features_scaled, columns=feature_names)
    return features_scaled, target, mu, sigma

def train_neural_net():
    '''
    Trains a neural net and saves it to a file

    '''
    raw_data = load_data()
    data = clean_data(raw_data)

    # Applying z-normalisation
    X, y, mu, sigma = standardise_data(data)
    data = pd.concat([X, y], axis=1)  # All the columns are stardardised with the exception of the target column (y)

    # Converting the data to a tensor
    # tensor_data = torch.from_numpy(np.array(data))
    tensor_data = TensorDataset(torch.from_numpy(np.array(X)), torch.from_numpy(np.array(y)))

    # Splitting the dataset into training, validation and testing subsets
    train_split = 1 # 60% of the data will be used for training
    valid_split = 0.2 # 20% of the data will be used for validation
    test_split = 0.1 # 20% of the data will be used for testing

    n_train = round(len(tensor_data)*train_split)
    n_valid = round(len(tensor_data)*valid_split)
    n_test = round(len(tensor_data)*test_split)

    n_unassigned = len(tensor_data) - (n_train + n_valid + n_test)
    n_train += n_unassigned   # any remaining samples not assigned is used for training

    train_set, valid_set, test_set = random_split(tensor_data, (n_train, n_valid, n_test))

    # setting up data loaders (for training, validation and testing)
    batch_size = 3
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    valid_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    # Creating the neural net
    input_nodes = 12  # Number of features
    hidden_nodes = 12 # Number of nodes of the hidden layer(s)
    output_nodes = 3  # Number of output classes

    myNet = NeuralNet_4H(input_nodes, hidden_nodes, output_nodes)

    # Setting up loss function, learning rate and optimizer (Stochastic Gradient Descent - SGD)
    learning_rate = 0.002
    optimiser = torch.optim.Adam(myNet.parameters(), lr=learning_rate)
    loss_criterion = nn.CrossEntropyLoss()

    # Training the neural net
    n_epochs = 300  # number of iterations over the entire dataset
    step = 0
    iteration_list = list()
    accuracy_data = list()
    loss_data = list()
    log_interval = 100 # how often we report accuracy and loss

    for epoch in range(n_epochs):
        for X_train, y_train in train_loader:
            step +=1
            X_train = Variable(X_train).float()
            y_train = Variable(y_train).long()

            # Resetting gradients
            optimiser.zero_grad()

            # Performing forward propagation
            y_predicted = myNet(X_train)

            # Calculating the Cross Entropy loss and gradients
            loss = loss_criterion(y_predicted, y_train)
            loss.backward()

            # Updating optimiser parameters
            optimiser.step()

            if step % log_interval == 0:
                # Checking the model with validation data
                correct_labels = 0
                total_labels = 0
                for X_valid, y_valid in valid_loader:
                    total_labels += len(y_valid)
                    X_valid = Variable(X_valid).float()
                    y_predicted = myNet(X_valid)
                    predictions = torch.max(y_predicted.data, dim=1)[1]   # classes with highest probability
                    # print('Predictions \n', predictions)
                    # print('Labels \n', y_valid)
                    for i in range(len(y_valid)):
                        if predictions[i].item() == y_valid[i].item():
                            correct_labels += 1

                    # correct_labels += (predictions.long() == y_valid.long()).sum()

                # Calculating Accuracy and storing loss and accuracy for plotting purposes
                accuracy = (correct_labels/total_labels)*100
                accuracy_data.append(accuracy)
                iteration_list.append(step)
                loss_data.append(loss.item())
                print(f'Epoch: {epoch} \t Loss: {loss.item()} \t Accuracy: {accuracy}')

    # Plotting results
    fig = plt.figure(figsize=(14, 6))
    ax1 = fig.add_subplot(121)
    plt.plot(iteration_list, accuracy_data, color = 'blue')
    plt.xlabel('Number of Steps')
    plt.ylabel('Accuracy')
    plt.ylim(0, 100)
    plt.xlim(left=0)

    ax2 = fig.add_subplot(122)
    plt.plot(iteration_list, loss_data, color = 'red')
    plt.xlabel('Number of Steps')
    plt.ylabel('Loss')
    plt.xlim(left=0)
    plt.ylim(bottom=0)
    plt.show()

    # Saving Neural net
    filename = os.path.join(os.path.dirname(__file__), 'trained_net.tar')
    torch.save({'model': myNet, 'mu': mu, 'sigma': sigma}, filename)
    return

def predict_thal()