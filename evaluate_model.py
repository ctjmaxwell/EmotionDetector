import keras
import numpy as np
from keras.utils import image_dataset_from_directory
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt

# 1. RELOAD THE DATA (We need the test data to grade the exam)
# Note: We don't need the 'train' folder here, just the 'test' folder.
IMG_SIZE = (48, 48)
BATCH_SIZE = 32

print("Loading Validation Data...")
val_ds = image_dataset_from_directory(
    directory='data/test',      # Make sure this path is correct!
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    label_mode='categorical',
    color_mode='grayscale',
    shuffle=False               # Important: Keep order so we can match predictions to truth
)

# 2. LOAD THE SAVED MODEL
# This skips the 30 epochs of training. It loads the result instantly.
print("Loading saved model...")
model = keras.models.load_model('emotion_detector_model.keras')

# 3. GENERATE PREDICTIONS
print("Generating predictions...")
y_true = []
y_pred = []

for images, labels in val_ds:
    # Get actual label index (e.g., 3 for Happy)
    y_true.extend(np.argmax(labels.numpy(), axis=1))
    
    # Get prediction
    predictions = model.predict(images, verbose=0)
    y_pred.extend(np.argmax(predictions, axis=1))

# 4. PLOT CONFUSION MATRIX
class_names = val_ds.class_names # e.g. ['Angry', 'Disgust', 'Fear'...]

plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_true, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names)
plt.ylabel('Actual Emotion')
plt.xlabel('Predicted Emotion')
plt.title('Confusion Matrix')
plt.show()

# 5. PRINT REPORT
print(classification_report(y_true, y_pred, target_names=class_names))