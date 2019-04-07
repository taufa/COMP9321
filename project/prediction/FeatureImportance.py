import pandas as pd
from project import db
from project.models import HeartDisease
import os
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
import numpy as np
from matplotlib import pyplot as plt
from sqlalchemy.orm import sessionmaker


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
    data.replace('?', np.nan, inplace=True)
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
    '''
    data = data.astype(np.float64)  # converting the entire dataframe to float64 data type
    scaler = StandardScaler()
    feature_names = data.columns[:-2]
    target = data.iloc[:, 12]
    features = data.iloc[:,:-2]   # the features are the rest of the columns
    features_scaled = scaler.fit_transform(features)
    features_scaled = pd.DataFrame(features_scaled, columns=feature_names)
    return features_scaled, target

def normalise_data(data):
    '''
    Normalises the data using min/max normalisation
    :param data: Dataframe containing the data
    :return: features_normalised: Dataframe containing the standardised features
             target: Dataframe containing the 'thal' column (target)
    '''
    feature_names = data.columns[:-2]
    target = data['thal']   # target
    features = data.iloc[:,:-2]   # the features are the rest of the columns
    features_normalised = normalize(features)
    features_normalised = pd.DataFrame(features_normalised, columns=feature_names)
    return features_normalised, target

def feature_chi2():
    '''
    Computes the Feature Importance using Chi squared.
    It assumes that the data is stored in a file called 'processed.cleveland.csv'
    :return: 'feature_importance' bar graph
    '''
    n_features = 12
    raw_data = load_data()
    data = clean_data(raw_data)

    features = data.iloc[:, :-2]
    target = data['thal']

    chi = SelectKBest(score_func=chi2, k=n_features)
    fit_result = chi.fit(features, target)

    feature_importance = pd.DataFrame(fit_result.scores_, columns=['contribution'])
    feature_names = pd.DataFrame(features.columns, columns=['features'])
    importance = pd.concat([feature_names, feature_importance], axis=1)
    importance = importance.sort_values(by='contribution', ascending=True)

    # Creating bar graph to show the feature importance
    fig = plt.figure(figsize=(10, 8))
    pos = np.arange(len(features.columns))
    plt.barh(pos, importance['contribution'], align='center', color='blue', alpha=0.5)
    plt.xlabel('Feature Contribution', fontsize=16)
    plt.title('Feature Importance using Chi squared', fontsize=20)
    plt.yticks(pos, importance['features'])
    plt.ylabel('Features', fontsize=16)
    #dir_path = os.path.dirname(os.path.realpath(__file__))
    dir_path = os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'static'))
    plt.savefig('{}\\img\\feature_importance.png'.format(dir_path), bbox_inches='tight')
    #plt.show()
    return
