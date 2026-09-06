"""
Diabetes Prediction — Flask Backend API
Loads trained ANN, ANFIS, scaler, and SHAP background to serve:
  GET  /api/results       — metrics, dataset summary, fuzzy rules, global SHAP importance
  POST /api/predict       — per-patient prediction from both models with SHAP explanation
  GET  /api/sample-patient/<idx> — return a real test patient's features (no dummy values)
"""

import os, json, warnings, traceback
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import shap

# -------------- Paths --------------
BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
DATA = os.path.join(BASE, "data", "pima_diabetes.csv")
FRONTEND = os.path.normpath(os.path.join(BASE, "..", "frontend"))

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
N_FEATURES = len(FEATURES)

SEED = 42


# -------------- Load artifacts --------------
print("Loading trained models...")

scaler = joblib.load(os.path.join(MODELS, "scaler.pkl"))
ann = keras.models.load_model(os.path.join(MODELS, "ann.keras"))

# Reconstruct ANFIS model (needs the same class definition as train.py)
class ANFIS(keras.Model):
    OUTPUT_SCALE = 5.0

    def __init__(self, n_features, n_rules, centers_init, sigmas_init, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.n_rules = n_rules
        self._centers_init = centers_init.astype(np.float32)
        self._sigmas_init  = sigmas_init.astype(np.float32)

    def build(self, input_shape):
        self.centers = self.add_weight(
            name="centers", shape=(self.n_rules, self.n_features),
            initializer=keras.initializers.Constant(self._centers_init), trainable=True)
        self.sigmas = self.add_weight(
            name="sigmas", shape=(self.n_rules, self.n_features),
            initializer=keras.initializers.Constant(self._sigmas_init), trainable=True)
        self.consequent_w = self.add_weight(
            name="consequent_w", shape=(self.n_rules, self.n_features),
            initializer=keras.initializers.Zeros(), trainable=True)
        self.consequent_b = self.add_weight(
            name="consequent_b", shape=(self.n_rules,),
            initializer=keras.initializers.Zeros(), trainable=True)
        super().build(input_shape)

    def call(self, x):
        # Returns LOGITS. Use predict_proba(x) for [0,1] probabilities.
        x_exp = tf.expand_dims(x, axis=1)
        c_exp = tf.expand_dims(self.centers, axis=0)
        s_exp = tf.expand_dims(tf.abs(self.sigmas) + 1e-4, axis=0)
        mu = tf.exp(-0.5 * tf.square((x_exp - c_exp) / s_exp))
        firing = tf.reduce_prod(mu, axis=2)
        norm = firing / (tf.reduce_sum(firing, axis=1, keepdims=True) + 1e-9)
        consequents = tf.matmul(x, self.consequent_w, transpose_b=True) + self.consequent_b
        raw = tf.reduce_sum(norm * consequents, axis=1, keepdims=True)
        return raw * self.OUTPUT_SCALE

    def predict_proba(self, x):
        return tf.sigmoid(self(x))

    def firing_strengths(self, x):
        x_exp = tf.expand_dims(x, axis=1)
        c_exp = tf.expand_dims(self.centers, axis=0)
        s_exp = tf.expand_dims(tf.abs(self.sigmas) + 1e-4, axis=0)
        mu = tf.exp(-0.5 * tf.square((x_exp - c_exp) / s_exp))
        firing = tf.reduce_prod(mu, axis=2)
        return (firing / (tf.reduce_sum(firing, axis=1, keepdims=True) + 1e-9)).numpy()


anfis_init = np.load(os.path.join(MODELS, "anfis_init.npz"))
N_RULES = anfis_init["centers"].shape[0]
anfis = ANFIS(N_FEATURES, N_RULES, anfis_init["centers"], anfis_init["sigmas"])
# Build by calling once, then load weights
_ = anfis(tf.zeros((1, N_FEATURES)))
anfis.load_weights(os.path.join(MODELS, "anfis.weights.h5"))

with open(os.path.join(MODELS, "results.json"), "r") as f:
    RESULTS = json.load(f)

shap_background = np.load(os.path.join(MODELS, "shap_background.npy"))
def ann_predict_proba(x):
    return ann.predict(x, verbose=0).flatten()
shap_explainer = shap.KernelExplainer(ann_predict_proba, shap_background)

# Load raw dataset for /api/sample-patient (real, not dummy)
df_raw = pd.read_csv(DATA)

print(f"Loaded ANN, ANFIS ({N_RULES} rules), scaler, SHAP explainer.")
print(f"Frontend dir: {FRONTEND}")


# -------------- App --------------
app = Flask(__name__, static_folder=None)
CORS(app)


@app.route("/api/results")
def get_results():
    """Return all training-time results: metrics, dataset stats, rules, importance."""
    return jsonify(RESULTS)


@app.route("/api/sample-patient/<int:idx>")
def get_sample_patient(idx):
    """Return a real patient from the dataset by index (no dummy data)."""
    if idx < 0 or idx >= len(df_raw):
        return jsonify({"error": "index out of range", "max": len(df_raw) - 1}), 400
    row = df_raw.iloc[idx]
    return jsonify({
        "index": int(idx),
        "features": {f: float(row[f]) for f in FEATURES},
        "actual_outcome": int(row["Outcome"]),
    })


@app.route("/api/predict", methods=["POST"])
def predict():
    """Predict from both ANN and ANFIS, with SHAP explanation."""
    try:
        body = request.get_json(force=True)
        # Build feature vector in fixed order
        try:
            x_raw = np.array([[float(body[f]) for f in FEATURES]], dtype=np.float32)
        except (KeyError, TypeError, ValueError) as e:
            return jsonify({"error": f"Missing or invalid feature: {e}"}), 400

        # Apply same preprocessing: replace 0s with median in clinical fields, then normalize
        ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
        medians = {f: RESULTS["dataset"]["feature_stats"][f]["median"] for f in FEATURES}
        x_clean = x_raw.copy()
        for col in ZERO_AS_MISSING:
            j = FEATURES.index(col)
            if x_clean[0, j] == 0:
                x_clean[0, j] = medians[col]
        x_scaled = scaler.transform(x_clean).astype(np.float32)

        # Predict (ANN already outputs probability; ANFIS outputs logits, so use predict_proba)
        ann_prob = float(ann.predict(x_scaled, verbose=0)[0, 0])
        anfis_prob = float(anfis.predict_proba(x_scaled).numpy()[0, 0])
        ann_label = int(ann_prob >= 0.5)
        anfis_label = int(anfis_prob >= 0.5)

        # SHAP for this single patient (against ANN)
        shap_vals = shap_explainer.shap_values(x_scaled, nsamples=80, silent=True)
        shap_arr = np.array(shap_vals)
        if shap_arr.ndim == 3:
            shap_arr = shap_arr[0]
        shap_per_feature = shap_arr[0].tolist() if shap_arr.ndim == 2 else shap_arr.tolist()
        base_value = float(shap_explainer.expected_value)
        if isinstance(base_value, (list, np.ndarray)):
            base_value = float(np.array(base_value).flatten()[0])

        explanation = [
            {"feature": FEATURES[i],
             "value": float(x_clean[0, i]),
             "shap": float(shap_per_feature[i])}
            for i in range(N_FEATURES)
        ]
        explanation.sort(key=lambda d: abs(d["shap"]), reverse=True)

        # ANFIS rule firing strengths for this patient — show top 3 rules
        firings = anfis.firing_strengths(x_scaled)[0]
        top_rule_idx = np.argsort(firings)[::-1][:3].tolist()
        top_rules = []
        all_rules = RESULTS["fuzzy_rules"]
        for r_idx in top_rule_idx:
            rule = all_rules[r_idx]
            top_rules.append({
                "rule_id": rule["rule_id"],
                "firing_strength": float(firings[r_idx]),
                "antecedents": rule["antecedents"],
                "conclusion": rule["conclusion"],
                "support_probability": rule["support_probability"],
            })

        return jsonify({
            "input_features": {f: float(x_clean[0, i]) for i, f in enumerate(FEATURES)},
            "preprocessing_note": "Zeros in clinical fields replaced with training-set medians.",
            "ann": {
                "probability": ann_prob,
                "prediction": ann_label,
                "label": "Diabetic" if ann_label else "Non-diabetic",
            },
            "anfis": {
                "probability": anfis_prob,
                "prediction": anfis_label,
                "label": "Diabetic" if anfis_label else "Non-diabetic",
            },
            "shap_explanation": {
                "base_value": base_value,
                "features": explanation,
            },
            "top_fuzzy_rules": top_rules,
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# -------------- Serve frontend --------------
@app.route("/")
def index():
    return send_from_directory(FRONTEND, "index.html")


@app.route("/<path:p>")
def static_proxy(p):
    return send_from_directory(FRONTEND, p)


if __name__ == "__main__":
    print("Starting Flask on http://0.0.0.0:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
