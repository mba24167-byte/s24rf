
import streamlit as st
import pandas as pd
import pickle
from features import feature_columns

# Load the trained model
with open('rf_clf.pkl', 'rb') as file:
    model = pickle.load(file)

st.title('Employee Attrition Prediction App')

st.write('Please enter the employee details to predict attrition.')

# Create input fields for each feature
input_data = {}
for feature in feature_columns:
    if feature in ['Age', 'DailyRate', 'DistanceFromHome', 'HourlyRate', 'MonthlyIncome', 'MonthlyRate', 'NumCompaniesWorked', 'PercentSalaryHike', 'TotalWorkingYears', 'TrainingTimesLastYear', 'YearsAtCompany', 'YearsInCurrentRole', 'YearsSinceLastPromotion', 'YearsWithCurrManager']:
        input_data[feature] = st.number_input(f'Enter {feature}', value=0)
    elif feature in ['Education', 'EnvironmentSatisfaction', 'JobInvolvement', 'JobLevel', 'JobSatisfaction', 'PerformanceRating', 'RelationshipSatisfaction', 'StockOptionLevel', 'WorkLifeBalance']:
        input_data[feature] = st.slider(f'Select {feature}', 1, 4, 1)
    elif feature in ['Gender', 'OverTime']:
        input_data[feature] = st.selectbox(f'Select {feature}', [0, 1])
    else:
        # Handle one-hot encoded features, assuming binary 0/1
        input_data[feature] = st.selectbox(f'Select {feature}', [0, 1])

# Predict button
if st.button('Predict Attrition'):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([input_data])

    # Ensure all feature columns are present in the input_df and in the correct order
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0 # Default to 0 for one-hot encoded features not selected

    input_df = input_df[feature_columns]

    # Make prediction
    prediction = model.predict(input_df)
    prediction_proba = model.predict_proba(input_df)

    st.subheader('Prediction Result:')
    if prediction[0] == 1:
        st.write("The employee is likely to attrite (Leave the company).")
    else:
        st.write("The employee is likely to stay (Not attrite).")

    st.write(f"Probability of Attrition: {prediction_proba[0][1]*100:.2f}%")
    st.write(f"Probability of Staying: {prediction_proba[0][0]*100:.2f}%")
