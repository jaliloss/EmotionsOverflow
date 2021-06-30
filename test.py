import numpy as np
from tensorflow import keras
import cv2
import os


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Create the model
model = keras.models.load_model('model/model_keras.h5')
# emotions will be displayed on your face from the webcam feed
model.load_weights('model/model_weights.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

emotion_dict = ['anger','contempt','disgust','fear','happy','sadness','surprise']

# start the webcam feed
cap = cv2.VideoCapture('vtest.mp4')
facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    if not ret:
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.cvtColor(roi_gray,cv2.COLOR_GRAY2RGB)
        cropped_img = cv2.resize(roi_gray, (48, 48))
        cropped_img = np.array([cropped_img])
        cropped_img = cropped_img.astype('float32')
        cropped_img = cropped_img / 255;
        cropped_img = cropped_img[0:1]
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x+20, y-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.imshow('Video', cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
