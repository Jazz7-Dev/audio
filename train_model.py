"""
Fine-Tune YAMNet for Audio Classification
==========================================
This script fine-tunes YAMNet on the ESC-50 dataset using transfer learning.
The base YAMNet model is used as a feature extractor (frozen), and custom
classification layers are trained on top.

Author: Devansh
"""

import os
import json
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import librosa
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import urllib.request
import zipfile

# Reduce TensorFlow verbosity
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# ============================================
# CONFIGURATION
# ============================================
SAMPLE_RATE = 16000
AUDIO_DURATION = 3  # seconds
NUM_SAMPLES = SAMPLE_RATE * AUDIO_DURATION
BATCH_SIZE = 32
EPOCHS = 20
MODEL_SAVE_PATH = "trained_model.keras"
HISTORY_SAVE_PATH = "training_history.json"

# ============================================
# DOWNLOAD ESC-50 DATASET
# ============================================
def download_esc50():
    """Download and extract ESC-50 dataset if not present."""
    dataset_dir = "ESC-50-master"
    
    if os.path.exists(dataset_dir):
        print("ESC-50 dataset already exists.")
        return dataset_dir
    
    print("Downloading ESC-50 dataset...")
    url = "https://github.com/karolpiczak/ESC-50/archive/master.zip"
    zip_path = "ESC-50.zip"
    
    urllib.request.urlretrieve(url, zip_path)
    
    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(".")
    
    os.remove(zip_path)
    print("Dataset ready!")
    return dataset_dir

# ============================================
# LOAD AND PREPROCESS AUDIO
# ============================================
def load_audio_file(file_path, target_sr=SAMPLE_RATE, duration=AUDIO_DURATION):
    """Load audio file and pad/truncate to fixed length."""
    try:
        audio, sr = librosa.load(file_path, sr=target_sr, duration=duration)
        
        # Pad or truncate to fixed length
        if len(audio) < NUM_SAMPLES:
            audio = np.pad(audio, (0, NUM_SAMPLES - len(audio)))
        else:
            audio = audio[:NUM_SAMPLES]
        
        return audio.astype(np.float32)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None

def prepare_dataset(dataset_dir):
    """Load ESC-50 dataset and prepare for training."""
    import pandas as pd
    
    audio_dir = os.path.join(dataset_dir, "audio")
    meta_path = os.path.join(dataset_dir, "meta", "esc50.csv")
    
    # Load metadata
    meta = pd.read_csv(meta_path)
    
    # Get unique class names
    class_names = sorted(meta['category'].unique())
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}
    
    print(f"Found {len(class_names)} classes: {class_names[:5]}...")
    
    X = []
    y = []
    
    print("Loading audio files...")
    for _, row in tqdm(meta.iterrows(), total=len(meta)):
        file_path = os.path.join(audio_dir, row['filename'])
        audio = load_audio_file(file_path)
        
        if audio is not None:
            X.append(audio)
            y.append(class_to_idx[row['category']])
    
    X = np.array(X)
    y = np.array(y)
    
    # Save class names for later use
    with open("class_names.json", "w") as f:
        json.dump(class_names, f)
    
    return X, y, class_names

# ============================================
# BUILD MODEL WITH YAMNET EMBEDDINGS
# ============================================
def build_model(num_classes):
    """Build transfer learning model using YAMNet as feature extractor."""
    print("Loading YAMNet model...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    
    # Create a wrapper to extract embeddings
    def yamnet_embeddings(waveform):
        scores, embeddings, spectrogram = yamnet_model(waveform)
        # Average embeddings across time frames
        return tf.reduce_mean(embeddings, axis=0)
    
    # Build the classification model
    inputs = tf.keras.layers.Input(shape=(NUM_SAMPLES,), dtype=tf.float32, name='audio_input')
    
    # Use Lambda layer to wrap YAMNet
    embeddings = tf.keras.layers.Lambda(
        lambda x: tf.map_fn(yamnet_embeddings, x, fn_output_signature=tf.TensorSpec(shape=(1024,), dtype=tf.float32)),
        name='yamnet_embeddings'
    )(inputs)
    
    # Custom classification layers (THESE ARE TRAINABLE)
    x = tf.keras.layers.Dense(512, activation='relu', name='dense_1')(embeddings)
    x = tf.keras.layers.BatchNormalization(name='bn_1')(x)
    x = tf.keras.layers.Dropout(0.4, name='dropout_1')(x)
    
    x = tf.keras.layers.Dense(256, activation='relu', name='dense_2')(x)
    x = tf.keras.layers.BatchNormalization(name='bn_2')(x)
    x = tf.keras.layers.Dropout(0.3, name='dropout_2')(x)
    
    x = tf.keras.layers.Dense(128, activation='relu', name='dense_3')(x)
    x = tf.keras.layers.Dropout(0.2, name='dropout_3')(x)
    
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax', name='output')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name='YAMNet_FineTuned')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================
# SIMPLIFIED APPROACH: PRE-EXTRACT EMBEDDINGS
# ============================================
def extract_embeddings(X, yamnet_model):
    """Pre-extract YAMNet embeddings for faster training."""
    print("Extracting YAMNet embeddings...")
    embeddings = []
    
    for audio in tqdm(X):
        scores, emb, spec = yamnet_model(audio)
        # Average across time frames
        avg_emb = np.mean(emb.numpy(), axis=0)
        embeddings.append(avg_emb)
    
    return np.array(embeddings)

def build_classifier(num_classes, input_dim=1024):
    """Build a simple classifier for pre-extracted embeddings."""
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,)),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ], name='Audio_Classifier')
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# ============================================
# MAIN TRAINING FUNCTION
# ============================================
def main():
    print("=" * 60)
    print("YAMNET FINE-TUNING FOR AUDIO CLASSIFICATION")
    print("=" * 60)
    
    # Step 1: Download dataset
    dataset_dir = download_esc50()
    
    # Step 2: Prepare dataset
    X, y, class_names = prepare_dataset(dataset_dir)
    print(f"\nDataset loaded: {X.shape[0]} samples, {len(class_names)} classes")
    
    # Step 3: Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Step 4: Load YAMNet and extract embeddings
    print("\nLoading YAMNet model from TensorFlow Hub...")
    yamnet_model = hub.load("https://tfhub.dev/google/yamnet/1")
    
    X_train_emb = extract_embeddings(X_train, yamnet_model)
    X_val_emb = extract_embeddings(X_val, yamnet_model)
    print(f"Embeddings shape: {X_train_emb.shape}")
    
    # Step 5: Build and train classifier
    print("\nBuilding classification model...")
    model = build_classifier(len(class_names))
    model.summary()
    
    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=5,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6
        )
    ]
    
    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    history = model.fit(
        X_train_emb, y_train,
        validation_data=(X_val_emb, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        verbose=1
    )
    
    # Step 6: Save model and history
    print("\nSaving trained model...")
    model.save(MODEL_SAVE_PATH)
    print(f"Model saved to: {MODEL_SAVE_PATH}")
    
    # Save training history
    history_dict = {
        'accuracy': [float(x) for x in history.history['accuracy']],
        'val_accuracy': [float(x) for x in history.history['val_accuracy']],
        'loss': [float(x) for x in history.history['loss']],
        'val_loss': [float(x) for x in history.history['val_loss']]
    }
    
    with open(HISTORY_SAVE_PATH, 'w') as f:
        json.dump(history_dict, f, indent=2)
    print(f"Training history saved to: {HISTORY_SAVE_PATH}")
    
    # Final results
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    final_acc = history.history['val_accuracy'][-1]
    print(f"Final Validation Accuracy: {final_acc:.2%}")
    print(f"\nFiles created:")
    print(f"  - {MODEL_SAVE_PATH}")
    print(f"  - {HISTORY_SAVE_PATH}")
    print(f"  - class_names.json")
    print("\nRun 'python plot_training.py' to generate training graphs!")

if __name__ == "__main__":
    main()
