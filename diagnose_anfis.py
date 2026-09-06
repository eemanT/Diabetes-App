"""Diagnose why ANFIS is squashed around 0.5."""
import os, warnings, json
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from tensorflow import keras

BASE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(BASE, "models")
DATA = os.path.join(BASE, "data", "pima_diabetes.csv")

FEATURES = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age"]
SEED = 42

class ANFIS(keras.Model):
    def __init__(self, n_features, n_rules, centers_init, sigmas_init,
                 consequent_w_init=None, consequent_b_init=None):
        super().__init__()
        self.n_features = n_features
        self.n_rules = n_rules
        self.centers = tf.Variable(centers_init.astype(np.float32), name="centers")
        self.sigmas  = tf.Variable(sigmas_init.astype(np.float32), name="sigmas")
        if consequent_w_init is not None:
            self.consequent_w = tf.Variable(consequent_w_init.astype(np.float32), name="consequent_w")
        else:
            self.consequent_w = tf.Variable(
                tf.random.normal([n_rules, n_features], stddev=0.1, seed=SEED),
                name="consequent_w")
        if consequent_b_init is not None:
            self.consequent_b = tf.Variable(consequent_b_init.astype(np.float32), name="consequent_b")
        else:
            self.consequent_b = tf.Variable(tf.zeros([n_rules]), name="consequent_b")
        self.output_scale = tf.Variable(5.0, trainable=False, name="output_scale", dtype=tf.float32)
    def call(self, x):
        x_exp = tf.expand_dims(x, axis=1)
        c_exp = tf.expand_dims(self.centers, axis=0)
        s_exp = tf.expand_dims(tf.abs(self.sigmas) + 1e-4, axis=0)
        mu = tf.exp(-0.5 * tf.square((x_exp - c_exp) / s_exp))
        firing = tf.reduce_prod(mu, axis=2)
        norm = firing / (tf.reduce_sum(firing, axis=1, keepdims=True) + 1e-9)
        consequents = tf.matmul(x, self.consequent_w, transpose_b=True) + self.consequent_b
        raw = tf.reduce_sum(norm * consequents, axis=1, keepdims=True)
        scaled = raw * self.output_scale
        return tf.sigmoid(scaled), firing, norm, scaled

# Load
scaler = joblib.load(os.path.join(MODELS, "scaler.pkl"))
init = np.load(os.path.join(MODELS, "anfis_init.npz"))
m = ANFIS(8, init["centers"].shape[0], init["centers"], init["sigmas"])
_ = m(tf.zeros((1, 8)))
m.load_weights(os.path.join(MODELS, "anfis.weights.h5"))
print(f"Loaded ANFIS. Learned output_scale = {float(m.output_scale.numpy()):.4f}")
print(f"Consequent_w  range: [{m.consequent_w.numpy().min():.3f}, {m.consequent_w.numpy().max():.3f}]")
print(f"Consequent_b  range: [{m.consequent_b.numpy().min():.3f}, {m.consequent_b.numpy().max():.3f}]")
print()

# Prepare test set the same way the API does
df = pd.read_csv(DATA)
ZERO_AS_MISSING = ["Glucose","BloodPressure","SkinThickness","Insulin","BMI"]
for c in ZERO_AS_MISSING:
    df[c] = df[c].replace(0, np.nan).fillna(df[c].median())
X = df[FEATURES].values
y = df["Outcome"].values
Xs = scaler.transform(X).astype(np.float32)

probs, firing, norm, raw = m(Xs)
probs = probs.numpy().flatten()
firing = firing.numpy()
norm = norm.numpy()
raw = raw.numpy().flatten()

print("=" * 55)
print("ANFIS PROBABILITY DISTRIBUTION")
print("=" * 55)
bins = [0, 0.2, 0.4, 0.45, 0.5, 0.55, 0.6, 0.8, 1.0]
hist, _ = np.histogram(probs, bins=bins)
for i in range(len(hist)):
    bar = "#" * int(40 * hist[i] / hist.max())
    print(f"  {bins[i]:.2f}-{bins[i+1]:.2f}: {hist[i]:4d}  {bar}")

stuck = ((probs >= 0.45) & (probs <= 0.55)).sum()
print(f"\nPatients with probability in [0.45, 0.55]: {stuck}/{len(probs)} ({100*stuck/len(probs):.1f}%)")

print("\n" + "=" * 55)
print("RULE-FIRING DIAGNOSTICS (the real culprit)")
print("=" * 55)
print(f"\nRaw firing strength (product of 8 Gaussians):")
print(f"  mean: {firing.mean():.2e}   max: {firing.max():.2e}   min: {firing.min():.2e}")
print(f"  Fraction of (sample,rule) firings < 1e-6: {(firing < 1e-6).mean()*100:.1f}%")

print(f"\nNormalized firing (after softmax-style division):")
print(f"  per-sample max:  mean={norm.max(axis=1).mean():.3f}  median={np.median(norm.max(axis=1)):.3f}")
print(f"  per-sample entropy: mean={(-norm*np.log(norm+1e-12)).sum(1).mean():.3f}  (uniform={np.log(norm.shape[1]):.3f})")
print("  -> If entropy is near uniform, all rules fire equally -> output averages -> ~0.5")

print(f"\nPre-sigmoid 'raw' score:")
print(f"  range: [{raw.min():.3f}, {raw.max():.3f}]   mean: {raw.mean():.3f}   std: {raw.std():.3f}")
print("  -> If |raw| stays near 0, sigmoid(raw) ~ 0.5 for everyone")
