import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

np.random.seed(42)

loan_id = np.random.randint(10001,20001,800)

gender = np.random.randint(0,2,800)
marrital_status = np.random.randint(0,2,800)
education = np.random.randint(0,2,800)
self_employed = np.random.randint(0,2,800)
applicant_income = np.random.normal(50,15,800).round(0)
coapplicant_income = np.random.normal(50,15,800).round(0)
total_income = applicant_income + coapplicant_income
loan_amount = np.random.randint(100000,500000,800)
loan_period = np.random.randint(1,4,800)
credit_history = np.random.randint(0,2,800) #0=good , 1=bad
credit_score = ((total_income < 80) & (credit_history != 0)).astype(int) #1=good,0=bad


df_prediction = pd.DataFrame({
    "Loan_ID" : loan_id,
    "Gender" : gender,
    "Marrital_Status" : marrital_status,
    "Education" : education,
    "Self_Employed":self_employed,
    "Applicant_Income": applicant_income,
    "Co-Applicant_Income":coapplicant_income,
    "Total_Amount": total_income,
    "Loan_Amount":loan_amount,
    "Loan_Period": loan_period,
    "Credit_history": credit_history,
    "Credit_Score":credit_score
})

df_prediction.to_csv("Loan_Approval.csv",index=False)

DATA_PATH = "Loan_Approval.csv"
MODEL_PATH = "loan_price_approval.pkl"
lr_path = "loan_amount_approve.pkl"


def loan():

    df=pd.read_csv("Loan_Approval.csv")

    x = df.drop(columns=["Credit_Score","Loan_ID"])
    y = df['Credit_Score']

    x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

    model = LogisticRegression()

    model.fit(x_train,y_train)
    y_pred = model.predict(x_test)

    lr = LinearRegression()

    x = df["Total_Amount"]
    x1 = sorted(x)
    Y= df["Loan_Amount"]
    Y1 = sorted(Y)

    x2 = np.array(x1)
    Y2 = np.array(Y1)

    x2 = x2.reshape(-1,1)
    Y2 = Y2.reshape(-1,1)

    lr.fit(x2,Y2)



    with open(lr_path,'wb') as g:
        pickle.dump(lr,g)
    print("Model trained for lr",lr_path)

    with open(MODEL_PATH,'wb') as f:
        pickle.dump(model,f)
    print("Model trained & saved to",MODEL_PATH)

if __name__ == "__main__":
    loan()


