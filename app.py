
from flask import Flask, render_template, request
import numpy as np
import joblib
import os

app = Flask(__name__)

# Load the trained SVM model and scaler
MODEL_PATH = "glass_svm_model.pkl"
SCALER_PATH = "scaler.pkl"

model = None
scaler = None

def load_model():
    global model, scaler

    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return True

    return False


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None
    glass_type = None

    if request.method == "POST":
        try:
            # Read the nine glass measurements from the form
            feature_names = [
                "RI", "Na", "Mg", "Al", "Si",
                "K", "Ca", "Ba", "Fe"
            ]

            features = [
                float(request.form[name])
                for name in feature_names
            ]

            if not np.all(np.isfinite(features)):
                raise ValueError("Please enter valid finite numbers.")

            # Ensure the trained model is available
            if model is None or scaler is None:
                if not load_model():
                    raise FileNotFoundError(
                        "Model files not found. Run train_model.py first."
                    )

            # Scale input and predict glass type
            input_data = np.array(features).reshape(1, -1)
            input_scaled = scaler.transform(input_data)

            result = model.predict(input_scaled)[0]
            prediction = f"Glass Type {result}"
            glass_type = result

        except ValueError as exc:
            error = str(exc) or "Please enter valid numeric values."
        except FileNotFoundError as exc:
            error = str(exc)
        except Exception:
            error = "Prediction failed. Please check the model files and input values."

    return render_template(
        "index.html",
        prediction=prediction,
        glass_type=glass_type,
        error=error
    )


if __name__ == "__main__":
    load_model()
    app.run(debug=True)
