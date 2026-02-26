import os
import sys
import pickle
import pandas as pd
from flask import Flask, render_template, request

app = Flask(__name__)

# 1) 让 Python 能 import 到项目根目录的 ecs171model3.py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))          # .../web
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))   # .../ecs171_project
sys.path.insert(0, PROJECT_ROOT)

import ecs171model3  # 里面有 PreprocessArtifacts + preprocess_transform

# 2) 关键补丁：preprocess.pkl 当初记录的是 __main__.PreprocessArtifacts
#    所以我们把这个类挂到当前运行的 __main__ (也就是 app.py) 上
import __main__
__main__.PreprocessArtifacts = ecs171model3.PreprocessArtifacts

# 3) 再加载 pkl
MODEL_DIR = os.path.join(PROJECT_ROOT, "outputs_model3")
model = pickle.load(open(os.path.join(MODEL_DIR, "model.pkl"), "rb"))
preprocess = pickle.load(open(os.path.join(MODEL_DIR, "preprocess.pkl"), "rb"))

@app.route("/")
def home():
    return render_template("index.html", prediction_text="")

@app.route("/predict", methods=["POST"])
def predict():
    input_dict = {
        "age": float(request.form["age"]),
        "sex": int(request.form["sex"]),
        "chest pain type": int(request.form["chest_pain"]),
        "resting bp s": float(request.form["rest_bp"]),
        "cholesterol": float(request.form["chol"]),
        "fasting blood sugar": int(request.form["fbs"]),
        "resting ecg": int(request.form["rest_ecg"]),
        "max heart rate": float(request.form["max_hr"]),
        "exercise angina": int(request.form["ex_ang"]),
        "oldpeak": float(request.form["oldpeak"]),
        "ST slope": int(request.form["st_slope"]),
    }

    df_input = pd.DataFrame([input_dict])
    df_processed = ecs171model3.preprocess_transform(df_input, preprocess, has_target=False)

    pred = int(model.predict(df_processed)[0])
    result = "Heart Disease (1)" if pred == 1 else "No Heart Disease (0)"
    return render_template("index.html", prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)
