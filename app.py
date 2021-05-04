import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler
import sklearn
from flask import Flask,render_template,request,jsonify
from flask_cors import CORS,cross_origin

app = Flask(__name__)
model = pickle.load(open('model.pkl','rb'))
@cross_origin()
@app.route('/',methods=['GET'])
def home():
    return render_template('index.html')
scaler= MinMaxScaler()
@cross_origin()
@app.route('/predict',methods=['POST','GET'])
def predict():
    if request.method == "POST":
        # Number of Pregnancies
        Pregnancies = float(request.form['Pregnancies'])
        Glucose = float(request.form['Glucose'])
        BloodPressure = float(request.form['BloodPressure'])
        SkinThickness = float(request.form['Skin Thickness'])
        Insulin = float(request.form['Insulin'])
        BMI = float(request.form['BMI'])
        DiabetesPedigreeFunction = float(request.form['DiabetesPedigreeFunction'])
        Age = float(request.form['Age'])
        feat = [[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]]
        #features = scaler.fit_transform(feat)

        prediction = model.predict(feat)
        output = prediction
        print(output)

        if output == 1:
            return render_template('index.html',prediction_text = "PERSON HAS DIABETES")
        else:
            return render_template('index.html',prediction_text = "PERSON DOESNOT HAS DIABETES")
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run()