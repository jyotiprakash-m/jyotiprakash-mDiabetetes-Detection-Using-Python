# Description: This program detects someone has diabetes or not by using mechine learning

# Import all the libraries
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import streamlit as st


# Creating title and subtitle
st.write("""
    # Diabetetes Detection
    Detect someone has diabetes or not

""")

image = Image.open('./cause_of_diabetes.png')

st.image(image, caption='Mechine Learning', use_column_width=True)

# get the data
df = pd.read_csv('./diabetes.csv')

# Set a subheader
st.subheader('Data Information')

st.dataframe(df)

st.write(df.describe())

# Show the data as a chart
chart = st.bar_chart(df)

# Split the data into independent 'X and dependent variable
X = df.iloc[:, 0:8].values
Y = df.iloc[:, -1].values

# split the data set into 75% tranning and 25% testing
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.25, random_state=0)

# get the feature input from the user


def get_user_input():
    Pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    Glucose = st.sidebar.slider('Glucose', 0, 199, 117)
    BloodPressure = st.sidebar.slider('BloodPressure', 0, 122, 72)
    SkinThickness = st.sidebar.slider('SkinThickness', 0, 99, 23)
    Insulin = st.sidebar.slider('Insulin', 0.0, 846.0, 30.0)
    BMI = st.sidebar.slider('BMI', 0.0, 67.1, 32.0)
    DiabetesPedigreeFunction = st.sidebar.slider(
        'DiabetesPedigreeFunction', 0.078, 2.42, 0.3725)
    Age = st.sidebar.slider('Age', 21, 81, 29)

    # Strore a dictionary into a variable
    user_data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age

    }

    # Train the data into a data frame
    features = pd.DataFrame(user_data, index=[0])
    return features


# Store the user input into a variable
user_input = get_user_input()

# Set a sub-header and diaplay the user input
st.subheader('User Input: ')
st.write(user_input)

# Create and train the model
RandomForestClassifier = RandomForestClassifier()
RandomForestClassifier.fit(X_train, Y_train)

# Show the models metrixs
st.subheader('Model test accuracy Score: ')
st.write(str(accuracy_score(Y_test, RandomForestClassifier.predict(X_test))*100)+'%')

# Store the models predections in a variable
prediction = RandomForestClassifier.predict(user_input)

# Set a subheader and display the classifier
st.subheader('Classification: ')
st.write(prediction)
