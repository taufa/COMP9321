import pandas as pd
#from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

#from sklearn.metrics import classification_report, confusion_matrix


# Declaration of classifiers as global variables
Classifier_SVC = None
Classifier_kNN = None
Classifier_DecisionTree = None
Classifier_LogisticRegression = None
Classifier_GausNB = None
Classifier_LDAnalysis = None

# Traning data can be any data located within the server path or external URL
training_data_fullpath = "processed.cleveland.data"

# Column names of the data in the training data file. 14 columns in total
col_names = ['age', 'sex', 'chest pain type', 'resting blood pressure', 'serum cholestoral', 'fasting blood sugar', 'resting electrocardiographic results', 'maximum heart rate achieved', 'exercise induced angina', 'oldpeak', 'the slope of the peak exercise ST segment', 'number of major vessels', 'thal', 'target']


# Init_and_Train() will load the training data, initialise each modals and train each models
# Init_and_Train() should be called once and only once before we can use any functions below for
#   prediction purpose.
def Init_and_Train(training_data_fullpath):
    try:
        df = pd.read_csv("processed.cleveland.data", names=col_names, na_values='?')
    except ValueError:
        pass
        # Ignore the errors from trying converting NaNs to float values. Just keep NaNs
        # in the newly produced Dataframe df.

    # Clean the Dataframe df. Drop any row which contains a NaN in any field
    df1 = df.dropna(axis=0, how='any')

    # Drop the 'thal' column, as it is deemed as the data we need to predict based on remaining 13 columns of data
    X0 = df1.drop('thal', axis=1)

    # Drop the 'target' column, as we are not allowed to use it according to the red notet, track 1 of the Assn3 specification.
    X1 = X0.drop('target', axis=1)

    y = df1['thal']

    # Initialise all of the classifiers
    # I will fine-tune a lot of parameters of each classifier to reach better prediction results
    # in the next fews days. But you guys can still seamlessly work on using my functions.
    global Classifier_SVC, Classifier_kNN, Classifier_DecisionTree, Classifier_LogisticRegression, Classifier_GausNB, Classifier_LDAnalysis
    Classifier_SVC = SVC(kernel='rbf', degree=3, gamma='auto')
    Classifier_kNN = KNeighborsClassifier()
    Classifier_DecisionTree = DecisionTreeClassifier()
    Classifier_LogisticRegression = LogisticRegression()
    Classifier_GausNB = GaussianNB()
    Classifier_LDAnalysis = LinearDiscriminantAnalysis()

    # Train each classifier using training data imported above
    Classifier_SVC.fit(X1, y)
    Classifier_kNN.fit(X1, y)
    Classifier_DecisionTree.fit(X1, y)
    Classifier_LogisticRegression.fit(X1, y)
    Classifier_GausNB.fit(X1, y)
    Classifier_LDAnalysis.fit(X1, y)

    # Init_and_Train() function returns nothing


# Below are six functions you can use to obtain the predicted results based on different classifiers
# Input arguments: 12 items we take as an user input on the webpage, in exactly the same order as given
#                   in the Assn3 specification PDF
#                   Note: all 12 items should be in float type before being pass as arguments to any of the
#                           6 functions below
# Return value: A single floating number indicating the predicted 'thal'
#               Return None if Init_and_Train() was not called before
def Predict_using_SVC(m_age, m_sex, m_chest_pain_type, m_resting_blood_pressure, m_serum_cholestoral,
                      m_fasting_blood_sugar, m_resting_electrocardiographic_results, m_maximum_heart_rate_achieved,
                      m_exercise_induced_angina, m_oldpeak, m_slope_peak_exercise_ST_segment,
                      m_number_of_major_vessels):
    if None == Classifier_SVC:
        return None
    m_Single_Input_Row = (m_age, m_sex, m_chest_pain_type, m_resting_blood_pressure, m_serum_cholestoral,
                      m_fasting_blood_sugar, m_resting_electrocardiographic_results, m_maximum_heart_rate_achieved,
                      m_exercise_induced_angina, m_oldpeak, m_slope_peak_exercise_ST_segment,
                      m_number_of_major_vessels)
    m_Pred_Result = Classifier_SVC.predict( ( m_Single_Input_Row, ) )
    return m_Pred_Result[0]




def Predict_using_kNN(m_age, m_sex, m_chest_pain_type, m_resting_blood_pressure, m_serum_cholestoral,
                      m_fasting_blood_sugar, m_resting_electrocardiographic_results, m_maximum_heart_rate_achieved,
                      m_exercise_induced_angina, m_oldpeak, m_slope_peak_exercise_ST_segment,
                      m_number_of_major_vessels):
    if None == Classifier_kNN:
        return None
    m_Single_Input_Row = (m_age, m_sex, m_chest_pain_type, m_resting_blood_pressure, m_serum_cholestoral,
                      m_fasting_blood_sugar, m_resting_electrocardiographic_results, m_maximum_heart_rate_achieved,
                      m_exercise_induced_angina, m_oldpeak, m_slope_peak_exercise_ST_segment,
                      m_number_of_major_vessels)
    m_Pred_Result = Classifier_kNN.predict( ( m_Single_Input_Row, ) )
    return m_Pred_Result[0]




def Predict_using_DecisionTree(m_age, m_sex, m_chest_pain_type, m_resting_blood_pressure, m_serum_cholestoral,
                      m_fasting_blood_sugar, m_resting_electrocardiographic_results, m_maximum_heart_rate_achieved,
                      m_exercise_induced_angina, m_oldpeak, m_slope_peak_exercise_ST_segment,
                      m_number_of_major_vessels):
    if None == Classifier_DecisionTree:
        return None
    m_Single_Input_Row = (m_age, m_sex, m_chest_pain_type, m_resting_blood_pressure, m_serum_cholestoral,
                      m_fasting_blood_sugar, m_resting_electrocardiographic_results, m_maximum_heart_rate_achieved,
                      m_exercise_induced_angina, m_oldpeak, m_slope_peak_exercise_ST_segment,
                      m_number_of_major_vessels)
    m_Pred_Result = Classifier_DecisionTree.predict( ( m_Single_Input_Row, ) )
    return m_Pred_Result[0]




def Predict_using_LogisticRegression(m_age, m_sex, m_chest_pain_type, m_resting_blood_pressure, m_serum_cholestoral,
                      m_fasting_blood_sugar, m_resting_electrocardiographic_results, m_maximum_heart_rate_achieved,
                      m_exercise_induced_angina, m_oldpeak, m_slope_peak_exercise_ST_segment,
                      m_number_of_major_vessels):
    if None == Classifier_LogisticRegression:
        return None
    m_Single_Input_Row = (m_age, m_sex, m_chest_pain_type, m_resting_blood_pressure, m_serum_cholestoral,
                      m_fasting_blood_sugar, m_resting_electrocardiographic_results, m_maximum_heart_rate_achieved,
                      m_exercise_induced_angina, m_oldpeak, m_slope_peak_exercise_ST_segment,
                      m_number_of_major_vessels)
    m_Pred_Result = Classifier_LogisticRegression.predict( ( m_Single_Input_Row, ) )
    return m_Pred_Result[0]




def Predict_using_GausNB(m_age, m_sex, m_chest_pain_type, m_resting_blood_pressure, m_serum_cholestoral,
                      m_fasting_blood_sugar, m_resting_electrocardiographic_results, m_maximum_heart_rate_achieved,
                      m_exercise_induced_angina, m_oldpeak, m_slope_peak_exercise_ST_segment,
                      m_number_of_major_vessels):
    if None == Classifier_GausNB:
        return None
    m_Single_Input_Row = (m_age, m_sex, m_chest_pain_type, m_resting_blood_pressure, m_serum_cholestoral,
                      m_fasting_blood_sugar, m_resting_electrocardiographic_results, m_maximum_heart_rate_achieved,
                      m_exercise_induced_angina, m_oldpeak, m_slope_peak_exercise_ST_segment,
                      m_number_of_major_vessels)
    m_Pred_Result = Classifier_GausNB.predict( ( m_Single_Input_Row, ) )
    return m_Pred_Result[0]




def Predict_using_LDAnalysis(m_age, m_sex, m_chest_pain_type, m_resting_blood_pressure, m_serum_cholestoral,
                      m_fasting_blood_sugar, m_resting_electrocardiographic_results, m_maximum_heart_rate_achieved,
                      m_exercise_induced_angina, m_oldpeak, m_slope_peak_exercise_ST_segment,
                      m_number_of_major_vessels):
    if None == Classifier_LDAnalysis:
        return None
    m_Single_Input_Row = (m_age, m_sex, m_chest_pain_type, m_resting_blood_pressure, m_serum_cholestoral,
                      m_fasting_blood_sugar, m_resting_electrocardiographic_results, m_maximum_heart_rate_achieved,
                      m_exercise_induced_angina, m_oldpeak, m_slope_peak_exercise_ST_segment,
                      m_number_of_major_vessels)
    m_Pred_Result = Classifier_LDAnalysis.predict( ( m_Single_Input_Row, ) )
    return m_Pred_Result[0]





# -- main() --
#Init_and_Train(training_data_fullpath)
#ret = Predict_using_SVC(42.0,1.0,4.0,136.0,315.0,0.0,0.0,125.0,1.0,1.9,2.0,0.0)
#print(ret)
#ret = Predict_using_kNN(42.0,1.0,4.0,136.0,315.0,0.0,0.0,125.0,1.0,1.9,2.0,0.0)
#print(ret)
#ret = Predict_using_DecisionTree(42.0,1.0,4.0,136.0,315.0,0.0,0.0,125.0,1.0,1.9,2.0,0.0)
#print(ret)
#ret = Predict_using_LogisticRegression(42.0,1.0,4.0,136.0,315.0,0.0,0.0,125.0,1.0,1.9,2.0,0.0)
#print(ret)
#ret = Predict_using_GausNB(42.0,1.0,4.0,136.0,315.0,0.0,0.0,125.0,1.0,1.9,2.0,0.0)
#print(ret)
#ret = Predict_using_LDAnalysis(42.0,1.0,4.0,136.0,315.0,0.0,0.0,125.0,1.0,1.9,2.0,0.0)
#print(ret)