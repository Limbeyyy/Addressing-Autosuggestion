# train_cnn.py
import os
import json
import numpy as np
import pandas as pd
import argparse
from tensorflow.keras import layers, models, callbacks

def load_char_map(path=r'C:\Users\Yukesh Dhakal\OneDrive\Documents\Desktop\kataho\prepare_data\python\artifacts\char_map.json'):
    with open(path, encoding='utf-8') as f:
        return json.load(f)

def load_labels(path=r'C:\Users\Yukesh Dhakal\OneDrive\Documents\Desktop\kataho\prepare_data\python\labels.txt'):
    with open(path, encoding='utf-8') as f:
        return [l.strip() for l in f if l.strip()]

def encode_text(s, char2i, max_len):
    arr = [char2i.get(c, 0) for c in s]
    if len(arr) < max_len:
        arr += [0]*(max_len - len(arr))
    else:
        arr = arr[:max_len]
    return arr

def build_model(vocab_size, max_len, num_classes, emb_dim=32):
    inp = layers.Input(shape=(max_len,), dtype='int32')
    x = layers.Embedding(input_dim=vocab_size+1, output_dim=emb_dim, input_length=max_len)(inp)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.Conv1D(128, 3, activation='relu', padding='same')(x)
    x = layers.GlobalMaxPool1D()(x)
    x = layers.Dense(128, activation='relu')(x)
    out = layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    model = models.Model(inp, out)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model

def main(args):
    os.makedirs('models', exist_ok=True)

    # Load training data
    df = pd.read_csv(r'C:\Users\Yukesh Dhakal\OneDrive\Documents\Desktop\kataho\prepare_data\data\train.csv')
    char2i = load_char_map()
    labels = load_labels()
    label2i = {l: i for i, l in enumerate(labels)}
    max_len = args.max_len

    # Encode inputs
    X = np.array([encode_text(s, char2i, max_len) for s in df['input'].astype(str)], dtype=np.int32)
    y = np.array([label2i[y] for y in df['target'].astype(str)], dtype=np.int32)

    # Build model
    model = build_model(len(char2i), max_len, len(labels), emb_dim=args.emb_dim)

    # Callbacks
    cb = [callbacks.EarlyStopping(patience=100, restore_best_weights=True)]

    # Train
    model.fit(X, y, epochs=100, batch_size=32, validation_split=0.1, callbacks=cb)

    # Save model in Keras 3 format
    model_file = 'models/cnn.keras'   # <-- save as .keras for Keras 3
    model.save(model_file, include_optimizer=False)

    # Save metadata
    meta = {'max_len': max_len, 'emb_dim': args.emb_dim}
    with open('models/meta.json', 'w', encoding='utf-8') as f:
        json.dump(meta, f)

    print(f'Saved Keras model to {model_file}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--max_len', type=int, default=12)
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--batch', type=int, default=128)
    args = parser.parse_args()
    main(args)