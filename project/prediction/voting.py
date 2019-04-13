from project.prediction.NeuralNet import predict_thal_nn
from project.prediction.Thal_ML_Pred import Predict_using_DecisionTree, Predict_using_kNN, Predict_using_SVC, Predict_using_LogisticRegression



voting = {0: 0, 1: 0, 2: 0}  # initialising voting dictionary (the keys of the dict are classes, values are votes)


prediction_NN = predict_thal_nn(m_age, m_sex, m_chest_pain_type, m_resting_blood_pressure, m_serum_cholesterol,
                      m_fasting_blood_sugar, m_resting_electrocardiographic_results, m_maximum_heart_rate_achieved,
                      m_exercise_induced_angina, m_oldpeak, m_slope_peak_exercise_ST_segment,
                      m_number_of_major_vessels)

prediction_SVM = Predict_using_SVC(m_age, m_sex, m_chest_pain_type, m_resting_blood_pressure, m_serum_cholestoral,
                      m_fasting_blood_sugar, m_resting_electrocardiographic_results, m_maximum_heart_rate_achieved,
                      m_exercise_induced_angina, m_oldpeak, m_slope_peak_exercise_ST_segment,
                      m_number_of_major_vessels)
prediction_kNN = Predict_using_kNN(m_age, m_sex, m_chest_pain_type, m_resting_blood_pressure, m_serum_cholestoral,
                      m_fasting_blood_sugar, m_resting_electrocardiographic_results, m_maximum_heart_rate_achieved,
                      m_exercise_induced_angina, m_oldpeak, m_slope_peak_exercise_ST_segment,
                      m_number_of_major_vessels)

prediction_DecisionTree = Predict_using_DecisionTree(m_age, m_sex, m_chest_pain_type, m_resting_blood_pressure, m_serum_cholestoral,
                               m_fasting_blood_sugar, m_resting_electrocardiographic_results,
                               m_maximum_heart_rate_achieved,
                               m_exercise_induced_angina, m_oldpeak, m_slope_peak_exercise_ST_segment,
                               m_number_of_major_vessels)

prediction_LogisticRegression = Predict_using_LogisticRegression(m_age, m_sex, m_chest_pain_type, m_resting_blood_pressure,
                        m_serum_cholestoral, m_fasting_blood_sugar, m_resting_electrocardiographic_results,
                        m_maximum_heart_rate_achieved, m_exercise_induced_angina, m_oldpeak, m_slope_peak_exercise_ST_segment,
                        m_number_of_major_vessels)
if result == 0 or result == 3:
  flash("Normal")
if result == 1 or result == 6:
  flash("Fixed Defect")
if result == 2 or result == 7:
  flash("Reversable Defect")

voting[int(prediction_NN)] += 2
voting[int(prediction_DecisionTree)] += 1.5
voting[int(prediction_kNN)] += 1.5
voting[int(prediction_SVM)] += 1
voting[int(prediction_LogisticRegression)] += 1

final_prediction, votes = sorted(voting.items(), key = lambda x: x[1], reverse = True)[0]