from flask import Flask, render_template, request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

app = Flask(__name__)


crime_data = pd.read_csv("test_crime_dataset.csv")


label_encoders = {}
for column in ["DayOfWeek", "NearPlace", "CrimeAgainst"]:
    label_encoders[column] = LabelEncoder()
    crime_data[column] = label_encoders[column].fit_transform(crime_data[column])

label_encoders['Offenses'] = LabelEncoder()
crime_data['Offenses'] = label_encoders['Offenses'].fit_transform(crime_data['Offenses'])


crime_data['Date'] = pd.to_datetime(crime_data['Date'])
crime_data['Year'] = crime_data['Date'].dt.year
crime_data['Month'] = crime_data['Date'].dt.month
crime_data['Day'] = crime_data['Date'].dt.day


X = crime_data[['DayOfWeek', 'NearPlace', 'Latitude', 'Longitude', 'CrimeAgainst', 'Year', 'Month', 'Day']]
y = crime_data['Offenses']
print(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@app.route('/')
def home():
    return render_template("login.html")

@app.route('/login', methods=['POST'])
def login():
    username = request.form.get("username")
    password = request.form.get("password")
    if username == "Vishnu" and password == "DXvishnu":
        return render_template("crime.html")
    return "Login Failed...."

@app.route('/prediction', methods=['POST'])
def predict():
   
    day_of_week = request.form['dayofweek']
    crime_against = request.form['crimeagainst']
    near_place = request.form['nearplace']
    location = request.form['location']
    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])
    year = int(request.form['year'])
    month = int(request.form['month'])
    day = int(request.form['day'])
    model_name = request.form['model']


    day_of_week_encoded = label_encoders['DayOfWeek'].transform([day_of_week])
    crime_against_encoded = label_encoders['CrimeAgainst'].transform([crime_against])
    near_place_encoded = label_encoders['NearPlace'].transform([near_place])

    if model_name == 'Decision Tree Classifier':
        model = DecisionTreeClassifier(random_state=42)
    elif model_name == 'Random Forest Classifier':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    else:
        return "Invalid model name"


    model.fit(X_train, y_train)

    prediction = model.predict([[day_of_week_encoded[0], near_place_encoded[0], latitude, longitude, crime_against_encoded[0], year, month, day]])

    predicted_offense = label_encoders['Offenses'].inverse_transform(prediction)

    return render_template("crime.html", prediction=predicted_offense[2])


app.run()
