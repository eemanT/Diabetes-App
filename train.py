"""
Diabetes Prediction — Training Pipeline
Implements all 4 phases from the project proposal:
  Phase 1: Data Preprocessing
  Phase 2: ANN (feed-forward, ReLU + Sigmoid)
  Phase 3: ANFIS (Takagi-Sugeno with Gaussian membership functions)
  Phase 4: Explainable AI (SHAP)
"""

import os, json, warnings, random
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_DETERMINISTIC_OPS"] = "1"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import skfuzzy as fuzz
import shap

# -------------- Reproducibility --------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
keras.utils.set_random_seed(SEED)

# -------------- Paths --------------
BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data", "pima_diabetes.csv")
MODELS = os.path.join(BASE, "models")
os.makedirs(MODELS, exist_ok=True)

FEATURES = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
            "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
N_FEATURES = len(FEATURES)


# ===================================================================
# PHASE 1: DATA PREPROCESSING
# ===================================================================
print("=" * 60)
print("PHASE 1: Data Preprocessing")
print("=" * 60)

df = pd.read_csv(DATA)
print(f"Raw dataset: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Class balance: {df['Outcome'].value_counts().to_dict()}")

# Replace biologically impossible zeros with NaN, then impute with median
ZERO_AS_MISSING = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
df_clean = df.copy()
for col in ZERO_AS_MISSING:
    n_zero = (df_clean[col] == 0).sum()
    df_clean[col] = df_clean[col].replace(0, np.nan)
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    print(f"  {col}: imputed {n_zero} zero values with median")

X = df_clean[FEATURES].values
y = df_clean["Outcome"].values

# Stratified 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=SEED, stratify=y
)

# Min-max normalization
scaler = MinMaxScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

print(f"Train: {X_train_s.shape}, Test: {X_test_s.shape}")

# Save raw feature stats for the UI (min, max, mean for slider defaults)
feature_stats = {}
for i, name in enumerate(FEATURES):
    col = df_clean[name]
    feature_stats[name] = {
        "min": float(col.min()),
        "max": float(col.max()),
        "mean": float(col.mean()),
        "median": float(col.median()),
        "std": float(col.std()),
    }


# ===================================================================
# PHASE 2: ANN — Feed-forward Neural Network (ReLU + Sigmoid)
# ===================================================================
print("\n" + "=" * 60)
print("PHASE 2: ANN Training")
print("=" * 60)

def build_ann(seed_offset=0):
    init = keras.initializers.GlorotUniform(seed=SEED + seed_offset)
    m = keras.Sequential([
        layers.Input(shape=(N_FEATURES,)),
        layers.Dense(24, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4),
                     kernel_initializer=init),
        layers.Dropout(0.2, seed=SEED + seed_offset),
        layers.Dense(12, activation="relu", kernel_initializer=init),
        layers.Dense(1, activation="sigmoid", kernel_initializer=init),
    ])
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
              loss="binary_crossentropy",
              metrics=["accuracy"])
    return m

# Mild class weighting (square root of imbalance)
n0, n1 = (y_train == 0).sum(), (y_train == 1).sum()
class_weight = {0: 1.0, 1: float(np.sqrt(n0/n1))}
print(f"Class weights (mild): {class_weight}")

# Train multiple seeds, keep the model with best validation accuracy
print("Training ANN with multiple seeds (ensemble-style selection)...")
best_ann, best_val, best_hist = None, -1.0, None
for trial in range(5):
    keras.utils.set_random_seed(SEED + trial * 100)
    m = build_ann(seed_offset=trial * 100)
    h = m.fit(
        X_train_s, y_train,
        validation_split=0.15,
        epochs=300, batch_size=16, verbose=0,
        class_weight=class_weight,
        callbacks=[keras.callbacks.EarlyStopping(patience=40, restore_best_weights=True,
                                                  monitor="val_accuracy", mode="max")]
    )
    val_acc = max(h.history["val_accuracy"])
    print(f"  Trial {trial+1}: best val_acc = {val_acc:.4f}, epochs = {len(h.history['loss'])}")
    if val_acc > best_val:
        best_val, best_ann, best_hist = val_acc, m, h

ann = best_ann
ann_hist = best_hist
print(f"Selected ANN with val_acc = {best_val:.4f}")
keras.utils.set_random_seed(SEED)  # reset for downstream


# ===================================================================
# PHASE 3: ANFIS — Takagi-Sugeno with Gaussian Membership Functions
# ===================================================================
print("\n" + "=" * 60)
print("PHASE 3: ANFIS Training")
print("=" * 60)

# Use Fuzzy C-Means to discover cluster centers, each cluster -> one fuzzy rule
N_RULES = 8

# FCM expects features-as-rows
cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
    X_train_s.T, c=N_RULES, m=2.0, error=1e-5, maxiter=500, seed=SEED
)
# cntr shape: (N_RULES, N_FEATURES) - cluster centers in normalized space

# Initial spread (sigma) per rule per feature
sigmas_init = np.zeros((N_RULES, N_FEATURES))
hard_assign = np.argmax(u, axis=0)
for r in range(N_RULES):
    members = X_train_s[hard_assign == r]
    if len(members) > 1:
        sigmas_init[r] = np.std(members, axis=0) + 0.05
    else:
        sigmas_init[r] = 0.3
sigmas_init = np.clip(sigmas_init, 0.2, 1.0)

# Warm-start each rule's consequents from logistic regression, BUT with per-rule
# random noise so rules differentiate from epoch 1 instead of all collapsing
# to the same LR prediction. Divided by output_scale so the scaled output
# matches the LR logit at init.
from sklearn.linear_model import LogisticRegression
lr_warm = LogisticRegression(max_iter=1000, random_state=SEED, C=1.0)
lr_warm.fit(X_train_s, y_train)
OUTPUT_SCALE_INIT = 5.0
_rng = np.random.RandomState(SEED)
_base_w = lr_warm.coef_[0] / OUTPUT_SCALE_INIT
_base_b = lr_warm.intercept_[0] / OUTPUT_SCALE_INIT
warm_w = np.tile(_base_w, (N_RULES, 1)) + _rng.normal(0, 0.15, (N_RULES, N_FEATURES))
warm_b = np.full(N_RULES, _base_b) + _rng.normal(0, 0.15, N_RULES)

print(f"Initialized {N_RULES} fuzzy rules via Fuzzy C-Means clustering")
print(f"Warm-start consequents from LR (train acc: {lr_warm.score(X_train_s, y_train):.3f})"
      f" + per-rule noise so rules differentiate")


class ANFIS(keras.Model):
    """Takagi-Sugeno ANFIS with Gaussian antecedents and linear consequents.

    Forward pass outputs LOGITS. Train with BinaryCrossentropy(from_logits=True)
    to avoid sigmoid-saturation pinning predictions to 0.5.

    Use `.predict_proba(x)` to get [0,1] probabilities.

    All variables are created via `self.add_weight()` so Keras 3 tracks them
    for save_weights/load_weights. Using bare `tf.Variable()` silently fails:
    Keras 3 doesn't track them and save_weights writes an empty file.
    """
    OUTPUT_SCALE = 5.0  # fixed pre-sigmoid temperature

    def __init__(self, n_features, n_rules, centers_init, sigmas_init,
                 consequent_w_init=None, consequent_b_init=None, **kwargs):
        super().__init__(**kwargs)
        self.n_features = n_features
        self.n_rules = n_rules
        # Stash initializers; actual weights are created in build().
        self._centers_init = centers_init.astype(np.float32)
        self._sigmas_init  = sigmas_init.astype(np.float32)
        self._cw_init = (None if consequent_w_init is None
                         else consequent_w_init.astype(np.float32))
        self._cb_init = (None if consequent_b_init is None
                         else consequent_b_init.astype(np.float32))

    def build(self, input_shape):
        self.centers = self.add_weight(
            name="centers", shape=(self.n_rules, self.n_features),
            initializer=keras.initializers.Constant(self._centers_init),
            trainable=True)
        self.sigmas = self.add_weight(
            name="sigmas", shape=(self.n_rules, self.n_features),
            initializer=keras.initializers.Constant(self._sigmas_init),
            trainable=True)
        cw_init = (keras.initializers.Constant(self._cw_init) if self._cw_init is not None
                   else keras.initializers.RandomNormal(stddev=0.3, seed=SEED))
        self.consequent_w = self.add_weight(
            name="consequent_w", shape=(self.n_rules, self.n_features),
            initializer=cw_init, trainable=True)
        cb_init = (keras.initializers.Constant(self._cb_init) if self._cb_init is not None
                   else keras.initializers.RandomNormal(stddev=0.3, seed=SEED + 1))
        self.consequent_b = self.add_weight(
            name="consequent_b", shape=(self.n_rules,),
            initializer=cb_init, trainable=True)
        super().build(input_shape)

    def call(self, x):
        x_exp = tf.expand_dims(x, axis=1)
        c_exp = tf.expand_dims(self.centers, axis=0)
        s_exp = tf.expand_dims(tf.abs(self.sigmas) + 1e-4, axis=0)
        mu = tf.exp(-0.5 * tf.square((x_exp - c_exp) / s_exp))
        firing = tf.reduce_prod(mu, axis=2)
        norm = firing / (tf.reduce_sum(firing, axis=1, keepdims=True) + 1e-9)
        consequents = tf.matmul(x, self.consequent_w, transpose_b=True) + self.consequent_b
        raw = tf.reduce_sum(norm * consequents, axis=1, keepdims=True)
        return raw * self.OUTPUT_SCALE  # logits

    def predict_proba(self, x):
        return tf.sigmoid(self(x))


# Train multiple ANFIS seeds, keep the best on validation
print("Training ANFIS with multiple seeds...")
best_anfis, best_anfis_val, best_anfis_hist = None, -1.0, None
for trial in range(3):
    keras.utils.set_random_seed(SEED + trial * 100)
    m = ANFIS(N_FEATURES, N_RULES, cntr, sigmas_init,
              consequent_w_init=warm_w, consequent_b_init=warm_b)
    m.compile(optimizer=keras.optimizers.Adam(learning_rate=0.005),
              loss=keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[keras.metrics.BinaryAccuracy(name="accuracy", threshold=0.0)])
    h = m.fit(
        X_train_s, y_train,
        validation_split=0.15,
        epochs=500, batch_size=16, verbose=0,
        class_weight=class_weight,   # same imbalance handling as the ANN
        callbacks=[
            keras.callbacks.EarlyStopping(patience=80, restore_best_weights=True,
                                          monitor="val_accuracy", mode="max"),
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=30, min_lr=1e-5),
        ]
    )
    val_acc = max(h.history["val_accuracy"])
    print(f"  Trial {trial+1}: best val_acc = {val_acc:.4f}, epochs = {len(h.history['loss'])}")
    if val_acc > best_anfis_val:
        best_anfis_val, best_anfis, best_anfis_hist = val_acc, m, h

anfis = best_anfis
anfis_hist = best_anfis_hist
print(f"Selected ANFIS with val_acc = {best_anfis_val:.4f}")
keras.utils.set_random_seed(SEED)


# ===================================================================
# EVALUATION
# ===================================================================
print("\n" + "=" * 60)
print("EVALUATION ON TEST SET")
print("=" * 60)

def evaluate(model, X, y, name, from_logits=False):
    raw = model.predict(X, verbose=0).flatten()
    probs = (1.0 / (1.0 + np.exp(-raw))) if from_logits else raw
    preds = (probs >= 0.5).astype(int)
    metrics = {
        "accuracy":  float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds)),
        "recall":    float(recall_score(y, preds)),
        "f1":        float(f1_score(y, preds)),
        "auc_roc":   float(roc_auc_score(y, probs)),
        "confusion_matrix": confusion_matrix(y, preds).tolist(),
    }
    # ROC curve points for plotting (sampled)
    fpr, tpr, _ = roc_curve(y, probs)
    idx = np.linspace(0, len(fpr) - 1, min(60, len(fpr))).astype(int)
    metrics["roc"] = {
        "fpr": fpr[idx].tolist(),
        "tpr": tpr[idx].tolist(),
    }
    print(f"\n{name}:")
    for k in ["accuracy", "precision", "recall", "f1", "auc_roc"]:
        print(f"  {k:10s}: {metrics[k]:.4f}")
    print(f"  confusion : {metrics['confusion_matrix']}")
    return metrics

ann_metrics = evaluate(ann, X_test_s, y_test, "ANN")
anfis_metrics = evaluate(anfis, X_test_s, y_test, "ANFIS", from_logits=True)

# Save training history (sample for plotting)
def history_summary(h, max_pts=80):
    n = len(h.history["loss"])
    idx = np.linspace(0, n - 1, min(max_pts, n)).astype(int)
    return {
        "loss":     [float(h.history["loss"][i]) for i in idx],
        "val_loss": [float(h.history["val_loss"][i]) for i in idx],
        "acc":      [float(h.history["accuracy"][i]) for i in idx],
        "val_acc":  [float(h.history["val_accuracy"][i]) for i in idx],
        "epochs":   [int(i) for i in idx],
        "total_epochs": n,
    }

ann_metrics["history"] = history_summary(ann_hist)
anfis_metrics["history"] = history_summary(anfis_hist)


# ===================================================================
# PHASE 4: EXPLAINABLE AI — SHAP on the ANN
# ===================================================================
print("\n" + "=" * 60)
print("PHASE 4: Explainable AI (SHAP)")
print("=" * 60)

# Use a background sample for KernelExplainer (smaller is faster)
background = shap.sample(X_train_s, 80, random_state=SEED)
def ann_predict_proba(x):
    return ann.predict(x, verbose=0).flatten()

explainer = shap.KernelExplainer(ann_predict_proba, background)

# Compute mean |SHAP| on a test subset to rank global feature importance
print("Computing global SHAP feature importance...")
shap_sample = X_test_s[:60]
shap_values = explainer.shap_values(shap_sample, nsamples=80, silent=True)
shap_arr = np.array(shap_values)
if shap_arr.ndim == 3:
    shap_arr = shap_arr[0]
mean_abs_shap = np.abs(shap_arr).mean(axis=0)
global_importance = [
    {"feature": FEATURES[i], "importance": float(mean_abs_shap[i])}
    for i in range(N_FEATURES)
]
global_importance.sort(key=lambda d: d["importance"], reverse=True)
print("Global feature importance (mean |SHAP|):")
for d in global_importance:
    print(f"  {d['feature']:25s}: {d['importance']:.4f}")


# ===================================================================
# EXTRACT INTERPRETABLE FUZZY RULES FROM ANFIS
# ===================================================================
print("\nExtracting human-readable fuzzy rules from ANFIS...")

def linguistic_label(normalized_center):
    """Map [0,1]-normalized value to a Low/Medium/High label."""
    if normalized_center < 0.33:
        return "LOW"
    elif normalized_center < 0.67:
        return "MEDIUM"
    return "HIGH"

centers_trained = anfis.centers.numpy()
consequent_w = anfis.consequent_w.numpy()
consequent_b = anfis.consequent_b.numpy()

# Estimate each rule's average output direction (diabetic-leaning vs not) on training data
rule_outputs = []
x_in = tf.constant(X_train_s.astype(np.float32))
x_exp = tf.expand_dims(x_in, axis=1)
c_exp = tf.expand_dims(anfis.centers, axis=0)
s_exp = tf.expand_dims(tf.abs(anfis.sigmaslib if False else anfis.sigmas) + 1e-4, axis=0) \
    if False else tf.expand_dims(tf.abs(anfis.sigmas) + 1e-4, axis=0)
mu = tf.exp(-0.5 * tf.square((x_exp - c_exp) / s_exp))
firing = tf.reduce_prod(mu, axis=2).numpy()
norm_firing = firing / (firing.sum(axis=1, keepdims=True) + 1e-9)
consequents = (X_train_s @ consequent_w.T) + consequent_b
# Per-rule average contribution direction (sigmoid of consequent on its dominant samples)
rules_extracted = []
output_scale = float(ANFIS.OUTPUT_SCALE)
for r in range(N_RULES):
    # Pick samples where this rule fires strongest (top 5%)
    top_idx = np.argsort(norm_firing[:, r])[-max(1, int(0.05 * len(X_train_s))):]
    avg_consequent = float(np.mean(consequents[top_idx, r]))
    # Match the model: sigmoid happens AFTER the output_scale temperature.
    avg_prob = float(1.0 / (1.0 + np.exp(-avg_consequent * output_scale)))
    conclusion = "HIGH risk of diabetes" if avg_prob > 0.5 else "LOW risk of diabetes"
    # Antecedents: feature labels at this rule's center
    antecedents = []
    for j, feat in enumerate(FEATURES):
        c_norm = float(centers_trained[r, j])
        antecedents.append({
            "feature": feat,
            "label": linguistic_label(c_norm),
            "center_normalized": c_norm,
        })
    rules_extracted.append({
        "rule_id": r + 1,
        "antecedents": antecedents,
        "conclusion": conclusion,
        "support_probability": avg_prob,
        "coverage_pct": float(100.0 * (norm_firing[:, r].sum() / len(X_train_s))),
    })

print(f"Extracted {len(rules_extracted)} fuzzy rules.")


# ===================================================================
# SAVE EVERYTHING
# ===================================================================
print("\n" + "=" * 60)
print("Saving models and artifacts...")
print("=" * 60)

import joblib
ann.save(os.path.join(MODELS, "ann.keras"))

# Debug: confirm ANFIS has trained weights before saving
print(f"\nANFIS weight check before save:")
print(f"  m.weights count: {len(anfis.weights)}")
for w in anfis.weights:
    arr = w.numpy()
    print(f"    {w.name}: shape={arr.shape}  range=[{arr.min():.3f}, {arr.max():.3f}]")

anfis.save_weights(os.path.join(MODELS, "anfis.weights.h5"))

# Re-load and verify
import h5py
with h5py.File(os.path.join(MODELS, "anfis.weights.h5"), "r") as f:
    print(f"After save: file has {len(f.get('vars', {}))} weight datasets")

# Save ANFIS init params so we can reconstruct
np.savez(os.path.join(MODELS, "anfis_init.npz"),
         centers=cntr, sigmas=sigmas_init)

joblib.dump(scaler, os.path.join(MODELS, "scaler.pkl"))

# Background data for SHAP (used by API at predict-time)
np.save(os.path.join(MODELS, "shap_background.npy"), background)

# Dataset summary for UI
dataset_summary = {
    "n_total": int(df.shape[0]),
    "n_train": int(X_train_s.shape[0]),
    "n_test":  int(X_test_s.shape[0]),
    "n_features": N_FEATURES,
    "features": FEATURES,
    "outcome_counts": {
        "0": int((df["Outcome"] == 0).sum()),
        "1": int((df["Outcome"] == 1).sum()),
    },
    "zero_imputation": {col: int((df[col] == 0).sum()) for col in ZERO_AS_MISSING},
    "feature_stats": feature_stats,
}

bundle = {
    "ann": ann_metrics,
    "anfis": anfis_metrics,
    "dataset": dataset_summary,
    "global_importance": global_importance,
    "fuzzy_rules": rules_extracted,
    "n_rules": N_RULES,
}
with open(os.path.join(MODELS, "results.json"), "w") as f:
    json.dump(bundle, f, indent=2)

print("\nSaved:")
for fn in sorted(os.listdir(MODELS)):
    fp = os.path.join(MODELS, fn)
    print(f"  {fn}: {os.path.getsize(fp)/1024:.1f} KB")
print("\n✓ Training pipeline complete.")
