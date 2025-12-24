# convert_gbdt_tflite.py
import os
import joblib
import lightgbm as lgb
import xgboost as xgb
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

# -------------------------
# Paths for models
# -------------------------
LIGHTGBM_MODEL_FILE = 'models/lightgbm.txt'
XGBOOST_MODEL_FILE = 'models/xgboost.joblib'
OUTPUT_DIR = 'models/'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -------------------------
# Helper function: LightGBM → TFLite
# -------------------------
def convert_lightgbm_tflite(model_file, tflite_out):
    print("\n=== Converting LightGBM model to TFLite ===")
    
    # Load LightGBM
    lgb_model = lgb.Booster(model_file=model_file)

    # Convert LightGBM → ONNX
    onnx_model = onnxmltools.convert_lightgbm(lgb_model, name="lgb_model")
    onnx_file = tflite_out.replace('.tflite', '.onnx')
    with open(onnx_file, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print("Saved LightGBM ONNX model:", onnx_file)

    # Convert ONNX → TensorFlow
    onnx_model_loaded = onnx.load(onnx_file)
    tf_rep = prepare(onnx_model_loaded)
    tf_model_path = tflite_out.replace('.tflite', '_tf')
    tf_rep.export_graph(tf_model_path)
    print("Exported TensorFlow model:", tf_model_path)

    # Convert TensorFlow → TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_out, 'wb') as f:
        f.write(tflite_model)
    print("Saved LightGBM TFLite model:", tflite_out)


# -------------------------
# Helper function: XGBoost → TFLite
# -------------------------
def convert_xgboost_tflite(model_file, tflite_out):
    print("\n=== Converting XGBoost model to TFLite ===")
    
    # Load XGBoost
    xgb_model = joblib.load(model_file)
    n_features = xgb_model.n_features_in_

    # Convert XGBoost → ONNX
    initial_type = [('input', FloatTensorType([None, n_features]))]
    onnx_model = onnxmltools.convert_xgboost(xgb_model, initial_types=initial_type)
    onnx_file = tflite_out.replace('.tflite', '.onnx')
    with open(onnx_file, 'wb') as f:
        f.write(onnx_model.SerializeToString())
    print("Saved XGBoost ONNX model:", onnx_file)

    # Convert ONNX → TensorFlow
    onnx_model_loaded = onnx.load(onnx_file)
    tf_rep = prepare(onnx_model_loaded)
    tf_model_path = tflite_out.replace('.tflite', '_tf')
    tf_rep.export_graph(tf_model_path)
    print("Exported TensorFlow model:", tf_model_path)

    # Convert TensorFlow → TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_out, 'wb') as f:
        f.write(tflite_model)
    print("Saved XGBoost TFLite model:", tflite_out)


# -------------------------
# Run conversions
# -------------------------
convert_lightgbm_tflite(LIGHTGBM_MODEL_FILE, os.path.join(OUTPUT_DIR, 'lightgbm.tflite'))
convert_xgboost_tflite(XGBOOST_MODEL_FILE, os.path.join(OUTPUT_DIR, 'xgboost.tflite'))

