import os
import sys
import json
import logging
import argparse
import numpy as np
import tensorflow as tf
from config import KERAS_FILE, TFLITE_OUT, REPR_FILE, CHAR_MAP, META_JSON

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def encode_line(s: str, char_map: dict, max_len: int) -> np.ndarray:
    """Encode a string into a fixed-length INT32 array using the char map."""
    arr = [char_map.get(c, 0) for c in s]
    arr = arr[:max_len] if len(arr) >= max_len else arr + [0] * (max_len - len(arr))
    return np.array(arr, dtype=np.int32)


def load_json(path: str, label: str) -> dict:
    """Load and return a JSON file, raising a clear error if missing."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"{label} not found: {path}")
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def load_repr_lines(path: str, limit: int) -> list[str]:
    """Load representative sample lines from a text file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Representative samples file not found: {path}")
    with open(path, encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    if not lines:
        raise ValueError(f"Representative samples file is empty: {path}")
    return lines[:limit]


def build_converter(model, char_map: dict, max_len: int, repr_limit: int, repr_lines: list[str]):
    """Configure and return a fully quantized TFLite converter."""
    repr_ds = [encode_line(s, char_map, max_len) for s in repr_lines[:repr_limit]]

    def representative_dataset():
        for arr in repr_ds:
            yield [arr.reshape(1, -1)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    return converter


def main(args):
    log.info("Loading Keras model from: %s", KERAS_FILE)
    if not os.path.exists(KERAS_FILE):
        raise FileNotFoundError(f"Keras model not found: {KERAS_FILE}")
    model = tf.keras.models.load_model(KERAS_FILE)

    char_map = load_json(CHAR_MAP, "Char map")
    meta = load_json(META_JSON, "Metadata")
    max_len = meta.get('max_len')
    if max_len is None:
        raise KeyError(f"'max_len' key missing in metadata: {META_JSON}")

    log.info("Loading representative samples (limit=%d) from: %s", args.repr_limit, REPR_FILE)
    repr_lines = load_repr_lines(REPR_FILE, args.repr_limit)
    log.info("Loaded %d representative samples", len(repr_lines))

    log.info("Configuring INT8 quantized TFLite converter ...")
    converter = build_converter(model, char_map, max_len, args.repr_limit, repr_lines)

    log.info("Converting model ...")
    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(TFLITE_OUT), exist_ok=True)
    with open(TFLITE_OUT, 'wb') as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    log.info("Saved TFLite model (%.1f KB) to: %s", size_kb, TFLITE_OUT)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Keras model to INT8 quantized TFLite')
    parser.add_argument('--repr_limit', type=int, default=500,
                        help='Max representative samples used for INT8 calibration (default: 500)')
    args = parser.parse_args()

    try:
        main(args)
    except (FileNotFoundError, KeyError, ValueError) as e:
        log.error("Configuration error: %s", e)
        sys.exit(1)
    except Exception as e:
        log.exception("Unexpected error during conversion: %s", e)
        sys.exit(2)