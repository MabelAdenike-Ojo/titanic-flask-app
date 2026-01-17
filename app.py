from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)

# Load model and training columns
model = joblib.load("titanic_model.joblib")
model_columns = joblib.load("titanic_columns.joblib")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get form data
        data = request.form.to_dict()
        input_df = pd.DataFrame([data])
        
        # Convert numeric columns to float
        numeric_cols = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare']
        for col in numeric_cols:
            input_df[col] = input_df[col].astype(float)
        
        # Convert categorical columns to match model dummies
        input_df = pd.get_dummies(input_df, columns=['Sex', 'Embarked'], drop_first=True)
        
        # Reindex to match training columns, fill missing with 0
        input_df = input_df.reindex(columns=model_columns, fill_value=0)
        
        # Make prediction
        prediction = model.predict(input_df)[0]
        result = "Survived" if prediction == 1 else "Not Survived"
        
        return render_template("index.html", prediction_text=f"Prediction: {result}")
    
    except Exception as e:
        return render_template("index.html", prediction_text=f"Error: {str(e)}")

if __name__ == "__main__":
    app.run(debug=False)