# convert_naive_bayes_onnx.py
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

# -----------------------------
# Paths
# -----------------------------
MODEL_FILE = "models/naive_bayes.joblib"
VECTORIZER_FILE = "models/vectorizer.joblib"
ONNX_OUT = "models/naive_bayes.onnx"

# -----------------------------
# Load model + vectorizer
# -----------------------------
clf = joblib.load(MODEL_FILE)
vec = joblib.load(VECTORIZER_FILE)
n_features = len(vec.feature_names_)

# -----------------------------
# Convert sklearn → ONNX
# -----------------------------
initial_type = [('input', FloatTensorType([None, n_features]))]
onnx_model = convert_sklearn(clf, initial_types=initial_type, target_opset=12)

# Save ONNX model
with open(ONNX_OUT, "wb") as f:
    f.write(onnx_model.SerializeToString())

print("✔ Saved ONNX model:", ONNX_OUT)
