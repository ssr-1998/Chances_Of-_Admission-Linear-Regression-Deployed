# This App Is Created By S.S.R
from flask import Flask, request, render_template
from flask_cors import cross_origin
from sklearn.preprocessing import StandardScaler
import pickle
import pandas as pd

app = Flask(__name__)


@app.route("/", methods=["GET"])
@cross_origin()
def homepage():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
@cross_origin()
def index():
    if request.method == "POST":
        try:
            """Reading the inputs written by the User."""
            gre_score = float(request.form["gre_score"])
            toefl_score = float(request.form["toefl_score"])
            university_rating = float(request.form["university_rating"])
            sop = float(request.form["sop"])
            lor = float(request.form["lor"])
            cgpa = float(request.form["cgpa"])
            is_research = request.form["research"]
            if is_research == "yes":
                research = 1
            else:
                research = 0

            """As we provided our model a Scaled Data. Therefore it will predict correctly on that kind of Data only.
                But we can't use transform function directly untill we use fit_transfrm function. For that here as well
            we will have to import our Data."""
            data = pd.read_csv("Admission_Prediction.csv")
            data['University Rating'] = data['University Rating'].fillna(data['University Rating'].mode()[0])
            data['TOEFL Score'] = data['TOEFL Score'].fillna(data['TOEFL Score'].mean())
            data['GRE Score'] = data['GRE Score'].fillna(data['GRE Score'].mean())
            data = data.drop(columns=['Serial No.'])
            x = data.drop(columns=['Chance of Admit'])

            filename = "finalized_model_self.pickle"
            model = pickle.load(open(filename, "rb"))
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(x)
            prediction = model.predict(scaler.transform([[gre_score, toefl_score, university_rating, sop, lor, cgpa, research]]))
            print("Prediction is", prediction[0]*100)
            return render_template("results.html", prediction=round(100*prediction[0]))
        except Exception as e:
            print("The Exception message is :", e)
            return "Something Is Wrong"
    else:
        return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
