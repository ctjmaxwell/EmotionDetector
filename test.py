import cv2
import numpy as np
import keras

# loading the trained model
print("Loading model...")
model = keras.models.load_model('emotion_detector_model.keras')

# array of emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# OpenCV face detector
face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

cap = cv2.VideoCapture(1)

print("Starting camera... Press 'q' to quit.")

while True:
    # Read a single frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale - model trained in grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)

        # Crop the face region
        roi_gray = gray[y:y+h, x:x+w]
        # Resize to 48x48 - size model has been trained on
        roi_gray = cv2.resize(roi_gray, (48, 48), interpolation=cv2.INTER_AREA)
        # Normalize pixels (0 to 1 range)
        roi_norm = roi_gray.astype('float32')

        # Add the extra dimensions for batch and channel -  The model expects shape (1, 48, 48, 1)
        roi_ready = np.expand_dims(roi_norm, axis=0)
        roi_ready = np.expand_dims(roi_ready, axis=-1)

        # make prediction
        prediction = model.predict(roi_ready, verbose=0)
        max_index = int(np.argmax(prediction))
        predicted_emotion = emotion_labels[max_index]

        # Display the emotion name on the screen
        cv2.putText(frame, predicted_emotion, (x, y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)


    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()