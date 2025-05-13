from flask import Flask,request,render_template
import pickle
import pandas as pd
import numpy as np

MODEL_PATH = "loan_price_approval.pkl"
lr_path = "loan_amount_approve.pkl"

with open(lr_path,'rb') as g:
    lr = pickle.load(g)


with open(MODEL_PATH,'rb') as f:
    model = pickle.load(f)

app = Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html",approve=None)

@app.route("/predict", methods= ["POST"])
def approve():

    gender = int(request.form['gender'])
    marrital_status = int(request.form['marrital_status'])
    education = int(request.form['Education'])
    self_employed = int(request.form['Self Employed'])
    applicant_income = float(request.form['Applicant Income'])
    coapplicant_income = float(request.form['Co-Applicant Income'])
    total_income = int(applicant_income) + int(coapplicant_income)
    loan_amount = float(request.form['Loan Amount'])
    loan_period = int(request.form['Loan Period(Yrs)'])
    credit_history = int(request.form['Credit History'])
    
    features = np.array([[gender, marrital_status, education, self_employed,
                          applicant_income, coapplicant_income, total_income,
                          loan_amount, loan_period, credit_history]])
    
   
    prediction = model.predict(features)[0]
    approve = lr.predict([[total_income]])
    
    result = ("Loan Approved ",approve) if prediction == 0 else "Loan Rejected"

    return f"<h3 style='text-align:center;'>{result}</h3><br><a href='/'>Back</a>"


if __name__== "__main__":
    app.run(debug=True)