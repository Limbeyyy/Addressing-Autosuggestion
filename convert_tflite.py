# convert_tflite.py
import tensorflow as tf
import numpy as np
import json
import os

# -------------------------
# Paths
# -------------------------
KERAS_MODEL_FILE = '../python/models/cnn_en.keras'  # .keras model file
TFLITE_OUT = '../python/models/model_en.tflite'     # Output TFLite file
REPR_FILE = '../python/python/labels_en.txt'
CHAR_MAP_FILE = '../python/python/artifacts/char_map_en.json'
META_FILE = '../python/models/meta_en.json'

# -------------------------
# Load Keras model
# -------------------------
model = tf.keras.models.load_model(KERAS_MODEL_FILE)

# -------------------------
# Prepare TFLite converter from Keras model
# -------------------------
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# -------------------------
# Load representation samples
# -------------------------
with open(REPR_FILE, encoding='utf-8') as f:
    repr_lines = [l.strip() for l in f if l.strip()]

# Load char map and max_len
with open(CHAR_MAP_FILE, encoding='utf-8') as f:
    char_map = json.load(f)
with open(META_FILE, encoding='utf-8') as f:
    meta = json.load(f)
max_len = meta['max_len']

# -------------------------
# Encode prefix string to integer array
# -------------------------
def encode_line(s):
    arr = [char_map.get(c, 0) for c in s]
    if len(arr) < max_len:
        arr += [0] * (max_len - len(arr))
    else:
        arr = arr[:max_len]
    return np.array(arr, dtype=np.int32)

# -------------------------
# Representative dataset for INT8 quantization
# -------------------------
repr_ds = [encode_line(s) for s in repr_lines[:500]]

def representative_dataset():
    for arr in repr_ds:
        yield [arr.reshape(1, -1)]

converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# -------------------------
# Convert and save TFLite model
# -------------------------
tflite_model = converter.convert()
os.makedirs(os.path.dirname(TFLITE_OUT), exist_ok=True)
with open(TFLITE_OUT, 'wb') as f:
    f.write(tflite_model)

print('Saved TFLite model to:', TFLITE_OUT)
