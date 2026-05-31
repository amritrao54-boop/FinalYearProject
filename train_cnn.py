"""
Train a CNN model for cattle disease image classification using transfer learning.

Usage:
    1. Create a dataset folder structure:
       data/cattle_images/
         ├── Healthy/
         ├── Foot_and_Mouth/
         ├── Mastitis/
         ├── Blackleg/
         ├── Foot_Rot/
         └── Lumpy_Skin/

    2. Place at least 20-30 images per class in each folder.
       You can download images from:
       - Kaggle: Search "cattle disease dataset"
       - Google Images: Search for each disease name

    3. Run this script:
       python train_cnn.py

    4. The trained model will be saved to data/cattle_disease_model.h5
       The app will automatically use it on next restart.
"""

import os
import sys
import json
import numpy as np

# ── Configuration ──
DATASET_DIR = os.path.join('data', 'cattle_images')
MODEL_SAVE_PATH = os.path.join('data', 'cattle_disease_model.h5')
CLASS_NAMES_PATH = os.path.join('data', 'class_names.json')
IMG_SIZE = (224, 224)
BATCH_SIZE = 16
EPOCHS = 15
VALIDATION_SPLIT = 0.2


def check_dataset():
    """Verify the dataset folder exists and has enough images."""
    if not os.path.exists(DATASET_DIR):
        print(f"\n[ERROR] Dataset folder not found at: {DATASET_DIR}")
        print(f"\nPlease create the following folder structure:")
        print(f"  {DATASET_DIR}/")
        print(f"    ├── Healthy/        (place 20+ cow images)")
        print(f"    ├── Foot_and_Mouth/ (place 20+ disease images)")
        print(f"    ├── Mastitis/       (place 20+ disease images)")
        print(f"    ├── Blackleg/       (place 20+ disease images)")
        print(f"    ├── Foot_Rot/       (place 20+ disease images)")
        print(f"    └── Lumpy_Skin/     (place 20+ disease images)")
        sys.exit(1)

    classes = [d for d in os.listdir(DATASET_DIR)
               if os.path.isdir(os.path.join(DATASET_DIR, d))]

    if len(classes) < 2:
        print(f"\n[ERROR] Need at least 2 class folders in {DATASET_DIR}")
        print(f"   Found: {classes}")
        sys.exit(1)

    print(f"\nDataset Summary:")
    print(f"   Location: {DATASET_DIR}")
    total = 0
    for cls in sorted(classes):
        cls_dir = os.path.join(DATASET_DIR, cls)
        count = len([f for f in os.listdir(cls_dir)
                     if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.webp'))])
        total += count
        status = "OK" if count >= 10 else "!!"
        print(f"   {status} {cls}: {count} images")

    print(f"   Total: {total} images across {len(classes)} classes\n")

    if total < 20:
        print("Warning: Very few images. Aim for 20+ per class for better accuracy.")

    return classes


def train():
    """Train the model using transfer learning with MobileNetV2."""
    print("Importing TensorFlow...")
    import tensorflow as tf
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
    from tensorflow.keras.models import Model
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    classes = check_dataset()

    # ── Data augmentation for training ──
    print("Loading and augmenting dataset...")
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        validation_split=VALIDATION_SPLIT,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.3,
        horizontal_flip=True,
        brightness_range=[0.7, 1.3],
        fill_mode='nearest'
    )

    train_gen = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )

    val_gen = train_datagen.flow_from_directory(
        DATASET_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )

    num_classes = len(train_gen.class_indices)
    class_names = {v: k for k, v in train_gen.class_indices.items()}

    print(f"\nBuilding model with {num_classes} classes...")
    print(f"   Classes: {list(train_gen.class_indices.keys())}")

    # ── Build transfer learning model ──
    base_model = MobileNetV2(
        weights='imagenet',
        include_top=False,
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
    )

    # Freeze base model layers initially
    base_model.trainable = False

    # Add custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    predictions = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    model.summary()

    # ── Callbacks ──
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7)
    ]

    # ── Phase 1: Train only the classification head ──
    print("\nPhase 1: Training classification head...")
    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )

    # ── Phase 2: Fine-tune the last layers of the base model ──
    print("\nPhase 2: Fine-tuning last 30 layers of MobileNetV2...")
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=10,
        callbacks=callbacks
    )

    # ── Evaluate ──
    val_loss, val_acc = model.evaluate(val_gen)
    print(f"\nFinal Validation Accuracy: {val_acc * 100:.1f}%")
    print(f"   Final Validation Loss: {val_loss:.4f}")

    # ── Save model and class names ──
    model.save(MODEL_SAVE_PATH)
    print(f"\nModel saved to: {MODEL_SAVE_PATH}")

    with open(CLASS_NAMES_PATH, 'w') as f:
        json.dump(class_names, f, indent=2)
    print(f"   Class names saved to: {CLASS_NAMES_PATH}")

    print(f"\nTraining complete! Restart the Flask app to use the new model.")
    print(f"   The app will automatically load {MODEL_SAVE_PATH}")


if __name__ == '__main__':
    train()
