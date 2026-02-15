import os
import keras
from keras import layers
from keras.utils import image_dataset_from_directory
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt

# image and batch size
IMG_SIZE = (48, 48)
BATCH_SIZE = 32

# Apply different weights to emotions since images numbers vary in size
class_weights = {
    0: 1.2,  # Angry
    1: 2.5,  # Disgust
    2: 1.2,  # Fear
    3: 1.0,  # Happy
    4: 1.0,  # Neutral
    5: 0.95,  # Sad
    6: 1.1   # Surprise
}

# training data
print("Loading Training Data...")
train_ds = image_dataset_from_directory(
    directory='data/train',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    color_mode='grayscale',
    shuffle=True
)

# validation data
print("Loading Validation Data...")
val_ds = image_dataset_from_directory(
    directory='data/test',
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    color_mode='grayscale',
    shuffle=False
)

# rescaling and rotating/zooming to augment data and prevent overfitting
print("Setting up Data Augmentation...")
data_augmentation = keras.Sequential([
    layers.Rescaling(1./255),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.3),
    layers.RandomFlip("horizontal"),
])

# building the model
model = keras.Sequential()

model.add(keras.Input(shape=(48, 48, 1)))
model.add(data_augmentation)

model.add(layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.2))

model.add(layers.Conv2D(128, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.3))

model.add(layers.Conv2D(256, kernel_size=(3, 3), padding='same', activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D(pool_size=(2, 2)))
model.add(layers.Dropout(0.4))

model.add(layers.Flatten())

model.add(layers.Dense(512, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.5))

model.add(layers.Dense(7, activation='softmax'))

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# DEFINE CALLBACKS (The Secret Weapon)
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=0.00001
)

# TRAIN WITH CALLBACKS
print("Starting Training with Callbacks...")
history = model.fit(
    train_ds,
    epochs=40,
    validation_data=val_ds,
    class_weight=class_weights,
    callbacks=[early_stopping, reduce_lr]
)

model.save('emotion_detector_model.keras')
print("Model saved successfully as emotion_detector_model.keras")

# Extract the data from the 'history' variable
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

# Create the list of epoch numbers (e.g., [1, 2, 3...])
epochs_range = range(len(acc))

# Setup the graph area
plt.figure(figsize=(14, 6))

# PLOT 1: Accuracy
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend(loc='lower right')
plt.grid(True) # Adds a grid to make it easier to read

# PLOT 2: Loss
plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend(loc='upper right')
plt.grid(True)

# Show the graphs
plt.show()