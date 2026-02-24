import os
import json
import numpy as np
import pandas as pd
import argparse
from tensorflow.keras import layers, models, callbacks
from config import LABELS, CHAR_MAP, TRAIN_CSV, KERAS_FILE, META_JSON


def load_char_map():
    """Load character-to-index mapping from config path."""
    with open(CHAR_MAP, encoding='utf-8') as f:
        return json.load(f)


def load_labels():
    """Load vocabulary label list from config path."""
    with open(LABELS, encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def encode_text(s, char2i, max_len):
    """Encode a string into a fixed-length integer array."""
    arr = [char2i.get(c, 0) for c in s]
    arr = arr[:max_len] if len(arr) >= max_len else arr + [0] * (max_len - len(arr))
    return arr


def build_model(vocab_size, max_len, num_classes, emb_dim=32):
    """Build and compile the CNN model."""
    inp = layers.Input(shape=(max_len,), dtype='int32')
    x = layers.Embedding(input_dim=vocab_size + 1, output_dim=emb_dim, input_length=max_len)(inp)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


def main(args):
    # Load data and mappings
    df = pd.read_csv(TRAIN_CSV)
    char2i = load_char_map()
    labels = load_labels()
    label2i = {label: i for i, label in enumerate(labels)}

    # Encode inputs and targets
    X = np.array([encode_text(s, char2i, args.max_len) for s in df['input'].astype(str)], dtype=np.int32)
    y = np.array([label2i[t] for t in df['target'].astype(str)], dtype=np.int32)

    # Build model
    model = build_model(len(char2i), args.max_len, len(labels), emb_dim=args.emb_dim)

    # Train
    cb = [callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
    model.fit(X, y, epochs=args.epochs, batch_size=args.batch, validation_split=0.1, callbacks=cb)

    # Save model and metadata
    os.makedirs(os.path.dirname(KERAS_FILE), exist_ok=True)
    model.save(KERAS_FILE, include_optimizer=False)

    meta = {'max_len': args.max_len, 'emb_dim': args.emb_dim}
    with open(META_JSON, 'w', encoding='utf-8') as f:
        json.dump(meta, f)

    print(f'Saved Keras model to: {KERAS_FILE}')
    print(f'Saved metadata to:    {META_JSON}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train CNN autosuggestion model')
    parser.add_argument('--max_len', type=int, default=12,  help='Max input sequence length')
    parser.add_argument('--emb_dim', type=int, default=32,  help='Embedding dimension')
    parser.add_argument('--epochs',  type=int, default=100, help='Max training epochs')
    parser.add_argument('--batch',   type=int, default=128, help='Batch size')
    args = parser.parse_args()
    main(args)