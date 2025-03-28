from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np

app = Flask(__name__, template_folder="C:/Users/anura/OneDrive/Desktop/python/disease/templates/index2.html")

# Load the trained model
model = joblib.load("disease_model.pkl")
symptom_list = joblib.load("symptom_list.pkl")

@app.route("/")
def home():
    return render_template("index.html")  # Loads frontend

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        symptoms = data["symptoms"].split(", ")  # Convert to list
        input_features = np.zeros(len(symptom_list))  # One-hot encoding

        for symptom in symptoms:
            if symptom in symptom_list:
                index = symptom_list.index(symptom)
                input_features[index] = 1  

        prediction = model.predict([input_features])[0]
        return jsonify({"predicted_disease": prediction})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
    