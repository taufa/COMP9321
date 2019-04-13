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
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)   # leaky relu activation function
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
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = torch.sigmoid(self.fc3(x))
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
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = torch.sigmoid(self.fc3(x))
        x = torch.tanh(self.fc4(x))
        output = F.log_softmax(self.fc5(x), dim=1)
        return output

class NeuralNet_5H(nn.Module):
    '''Custom Neural Network with 4 Hidden layers'''
    def __init__(self, input_nodes, hidden_nodes, output_nodes):
        super(NeuralNet_5H, self).__init__()
        self.fc1 = nn.Linear(input_nodes, hidden_nodes)
        self.fc2 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc3 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc4 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc5 = nn.Linear(hidden_nodes, hidden_nodes)
        self.fc6 = nn.Linear(hidden_nodes, output_nodes)

    def forward(self, x):
        x = torch.tanh(self.fc1(x))          # Using hyperbolic tangent as activation function
        x = torch.sigmoid(self.fc2(x))
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        x = torch.tanh(self.fc4(x))
        x = F.leaky_relu(self.fc5(x), negative_slope=0.2)
        output = F.log_softmax(self.fc6(x), dim=1)
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
    # Dropping the 2 least important features
    # data.drop(['fbs', 'restecg'], axis=1, inplace=True)
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
    # target = data.iloc[:, 12]
    features = data.iloc[:,:-2]   # the features are the rest of the columns
    mu = features.mean()
    sigma = features.std()
    features_scaled = scaler.fit_transform(features)
    features_scaled = pd.DataFrame(features_scaled, columns=feature_names)
    return features_scaled, target, mu, sigma

def train_neural_net(X, y, n_hidden_layers, n_hidden_nodes, max_epochs, batch_size, learning_rate,
                     weight_decay, train_acc, valid_acc):
    '''
    Trains a neural net and saves it to a file

    '''
    # raw_data = load_data()
    # data = clean_data(raw_data)

    # Applying z-normalisation
    # X, y, mu, sigma = standardise_data(data)
    # data = pd.concat([X, y], axis=1)  # All the columns are stardardised with the exception of the target column (y)

    # Converting the data to a tensor
    # tensor_data = torch.from_numpy(np.array(data))
    tensor_data = TensorDataset(torch.from_numpy(np.array(X)), torch.from_numpy(np.array(y)))

    # Splitting the dataset into training, validation and testing subsets
    train_split = 0.75 # 60% of the data will be used for training
    valid_split = 0.25 # 20% of the data will be used for validation
    test_split = 0 # 20% of the data will be used for testing

    n_train = round(len(tensor_data)*train_split)
    n_valid = round(len(tensor_data)*valid_split)
    n_test = round(len(tensor_data)*test_split)

    n_unassigned = len(tensor_data) - (n_train + n_valid + n_test)
    n_train += n_unassigned   # any remaining samples not assigned is used for training

    train_set, valid_set, test_set = random_split(tensor_data, (n_train, n_valid, n_test))

    # setting up data loaders (for training, validation and testing)
    batch_size_training = batch_size  # For training (mini-batches)
    batch_size_validation = n_valid
    train_loader = DataLoader(train_set, batch_size=batch_size_training, shuffle=False)
    valid_loader = DataLoader(valid_set, batch_size=batch_size_validation, shuffle=False)
    test_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False)

    # Creating the neural net
    input_nodes = 12  # Number of features
    hidden_nodes = n_hidden_nodes # Number of nodes of the hidden layer(s)
    output_nodes = 3  # Number of output classes

    if n_hidden_layers == 1:
        myNet = NeuralNet_1H(input_nodes, hidden_nodes, output_nodes)
    elif n_hidden_layers == 2:
        myNet = NeuralNet_2H(input_nodes, hidden_nodes, output_nodes)
    elif n_hidden_layers == 3:
        myNet = NeuralNet_3H(input_nodes, hidden_nodes, output_nodes)
    elif n_hidden_layers == 4:
        myNet = NeuralNet_4H(input_nodes, hidden_nodes, output_nodes)
    elif n_hidden_layers == 5:
        myNet = NeuralNet_5H(input_nodes, hidden_nodes, output_nodes)


    # Setting up loss function, learning rate and optimizer (Stochastic Gradient Descent - SGD)

    optimiser = torch.optim.Adam(myNet.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # loss_criterion = nn.MSELoss()
    loss_criterion = nn.CrossEntropyLoss()

    # Training the neural net
    n_epochs = max_epochs  # number of iterations over the entire training dataset
    step = 0
    iteration_list = list()
    valid_accuracy_list = list()
    train_accuracy_list = list()
    train_loss_list = list()
    valid_loss_list = list()
    log_interval = round(n_train/(10*batch_size)) # how often we report accuracy and loss
    train_accuracy = 0
    for epoch in range(n_epochs):
        correct_labels_train = 0
        total_labels_train = 0
        for X_train, y_train in train_loader:
            total_labels_train += len(y_train)
            step += 1
            X_train = Variable(X_train).float()
            y_train = Variable(y_train).long()

            # Resetting gradients
            optimiser.zero_grad()

            # Performing forward propagation
            y_predicted = myNet(X_train)

            # Calculate training accuracy
            predictions = torch.max(y_predicted.data, dim=1)[1]
            for i in range(len(y_train)):
                if predictions[i].item() == y_train[i].item():
                    correct_labels_train += 1

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
                    loss_valid = loss_criterion(y_predicted, y_valid.long())
                    predictions = torch.max(y_predicted.data, dim=1)[1]   # classes with highest probability
                    # print('Predictions \n', predictions)
                    # print('Labels \n', y_valid)
                    for i in range(len(y_valid)):
                        if predictions[i].item() == y_valid[i].item():
                            correct_labels += 1
                    # correct_labels += (predictions.long() == y_valid.long()).sum()

                # Calculating Accuracy and storing loss and accuracy for plotting purposes
                valid_accuracy = (correct_labels/total_labels)*100
                valid_accuracy_list.append(valid_accuracy)
                train_accuracy_list.append(train_accuracy)
                iteration_list.append(step)
                train_loss_list.append(loss.item())   # List of train loss data
                valid_loss_list.append(loss_valid.item())  # List of validation loss data
                # print('Epoch: {:d} \t Train Loss: {:.6f} \t Train Acc: {:.6f} \t Valid Loss: {:.6f} \t Validation Acc: {:.6f}'.\
                #       format(epoch, loss.item(), train_accuracy, loss_valid.item(), valid_accuracy))
        train_accuracy = (correct_labels_train / total_labels_train) * 100  # Accuracy is calculated after every epoch
        if ((train_accuracy > train_acc * 100) and (valid_accuracy > valid_acc * 100)) or (train_accuracy >= 95):
            # print('Stopping training early')
            break
    return myNet, iteration_list, train_accuracy_list, valid_accuracy_list, train_loss_list, valid_loss_list, train_accuracy, \
           valid_accuracy, epoch


def predict_thal_nn(m_age, m_sex, m_chest_pain_type, m_resting_blood_pressure, m_serum_cholesterol,
                      m_fasting_blood_sugar, m_resting_electrocardiographic_results, m_maximum_heart_rate_achieved,
                      m_exercise_induced_angina, m_oldpeak, m_slope_peak_exercise_ST_segment,
                      m_number_of_major_vessels):
    ''' Function that predicts a 'thal' value using the trained neural net stored in trained_net.tar'''
    X = [m_age, m_sex, m_chest_pain_type, m_resting_blood_pressure, m_serum_cholesterol,
                      m_fasting_blood_sugar, m_resting_electrocardiographic_results, m_maximum_heart_rate_achieved,
                      m_exercise_induced_angina, m_oldpeak, m_slope_peak_exercise_ST_segment,
                      m_number_of_major_vessels]
    # X = [m_age, m_sex, m_chest_pain_type, m_resting_blood_pressure, m_serum_cholesterol,
    #                   m_maximum_heart_rate_achieved, m_exercise_induced_angina, m_oldpeak, m_slope_peak_exercise_ST_segment,
    #                   m_number_of_major_vessels]

    filename = os.path.join(os.path.dirname(__file__), 'trained_net_14303.tar')

    # Loading the model from file
    model = torch.load(filename)
    myNet = model['model']
    myNet.eval()
    mu = model['mu']
    sigma = model['sigma']

    # Standardisation of inputs using the means and std devs stored in the model
    for i in range(len(X)):
        X[i] = (X[i] - mu[i])/sigma[i]
    X = torch.from_numpy(np.array([X]))
    X = Variable(X).float()

    # Run the model on the inputs
    y_predicted = myNet(X)
    prediction = torch.max(y_predicted.data, dim=1)[1]  # classes with highest probability
    return prediction.item()
# 63,1,1,145,233,1,2,150,0,2.3,3,0
# 60,1,4,130,206,0,2,132,1,2.4,2,2
# 58,0,3,120,340,0,0,172,0,0,1,0

def find_best_neural_net(starting_model_number, number_of_hidden_layers):
    ''' This function trains many different neural nets with different configurations with many different parameters
    and stores the models with the best accuracy'''
    raw_data = load_data()
    data = clean_data(raw_data)
    X, y, mu, sigma = standardise_data(data)

    # Creating the records dataframe that will store the parameters used for successful models
    records = pd.DataFrame(columns=['model', 'hidden_layers', 'hidden_nodes', 'epochs', 'batch_size', 'learning_rate',
                                    'weight_decay', 'train_acc', 'valid_acc'])

    # records = pd.read_csv(f'records_{number_of_hidden_layers}.csv')
    n_hidden_nodes = [6, 12, 24, 36]   # number of hidden nodes that will be tested
    max_epochs = 1500
    batch_sizes = [4, 8, 16, 32]
    learning_rates = [0.0002, 0.0004, 0.0008, 0.001, 0.002, 0.004, 0.008, 0.01, 0.02,
                      0.04, 0.08, 0.1]
    weight_decays = [0, 1e-6, 1e-5, 2e-5]
    model_number = starting_model_number
    # The model will automatically stop training once it reaches train_acc and valid_acc
    train_acc = 0.90
    valid_acc = 0.75

    for hl in [number_of_hidden_layers]:   # number of hidden layers range(1,5)
        for hn in n_hidden_nodes:  # number of hidden nodes
            for bs in batch_sizes:
                for lr in learning_rates:
                    for wd in weight_decays:
                        for i in range(5):
                            model_number += 1
                            print(f'Hidden layers: {hl}\t Hidden nodes: {hn}\t Batch size: {bs}\t Learning Rate: {lr}\t Weight Decay: {wd}\t Attempt {i}')
                            model, iteration_list, train_accuracy_list, valid_accuracy_list, train_loss_list, valid_loss_list, \
                            train_accuracy, valid_accuracy, epoch = train_neural_net(X, y, hl, hn, max_epochs, bs, lr,
                                                                                     wd, train_acc, valid_acc)
                            if valid_accuracy >= 75:
                                print('Model found')
                                records = records.append(
                                    {'model': model_number, 'hidden_layers': hl, 'hidden_nodes': hn, 'epochs': epoch,
                                     'batch_size': bs, 'learning_rate': lr, 'weight_decay': wd, 'train_acc': train_accuracy,
                                     'valid_acc': valid_accuracy},
                                    ignore_index=True)
                                # Save the records to file
                                records.to_csv(f'records_{number_of_hidden_layers}.csv', index=False, encoding='utf-8')

                                # Plotting results
                                fig = plt.figure(figsize=(14, 6))
                                ax1 = fig.add_subplot(121)
                                plt.plot(iteration_list, train_accuracy_list, color='red')
                                plt.plot(iteration_list, valid_accuracy_list, color='blue')
                                plt.legend(('Training', 'Validation'), loc='lower right')
                                plt.xlabel('Number of Iterations')
                                plt.ylabel('Accuracy')
                                plt.title('Training and Validation Accuracy vs Number of Iterations')
                                plt.ylim(0, 100)
                                plt.xlim(left=0)

                                ax2 = fig.add_subplot(122)
                                plt.plot(iteration_list, train_loss_list, color='red')
                                plt.plot(iteration_list, valid_loss_list, color='blue')
                                plt.legend(('Training', 'Validation'), loc='upper left')
                                plt.xlabel('Number of Iterations')
                                plt.ylabel('Loss')
                                plt.title('Training and Validation Loss vs Number of Iterations')
                                plt.xlim(left=0)
                                plt.ylim(bottom=0)
                                plt.suptitle(f'Hidden Layers: {hl}, Hidden Nodes:{hn}, Batch Size: {bs}, Learning Rate: {lr}, Weight Decay: {wd}')

                                # Save figure and model if the validation accuracy is greater than 75%
                                plt.savefig(f'Plot_{model_number}.png')
                                plt.close(fig)
                                # plt.show()

                                # Saving Neural net
                                filename = os.path.join(os.path.dirname(__file__), f'trained_net_{model_number}.tar')
                                torch.save({'model': model, 'mu': mu, 'sigma': sigma}, filename)
    return
