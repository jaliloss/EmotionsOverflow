import numpy as np
from tensorflow import keras
import cv2
import os
from time import sleep

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# Create the model
model = keras.models.load_model('model/model_keras.h5')
# emotions will be displayed on your face from the webcam feed
model.load_weights('model/model_weights.h5')

# prevents openCL usage and unnecessary logging messages
cv2.ocl.setUseOpenCL(False)

emotion_dict = ['Anger','Contempt','Disgust','Fear','Happy','Sadness','Surprise']

test_vid_dir = 'OuluCASIA'

#N to move to next emotion
#P to move to previous emotion
#S to skip the person and move to next person
#R to return to previous person
#Q to quit

def load_image(img_path):
  input_img=cv2.imread(img_path)
  return input_img


for person in os.listdir(test_vid_dir):
    person_dir = test_vid_dir + '/' + person
    for emotion in os.listdir(person_dir):
        emotion_dir = person_dir + '/' + emotion
        emotion_frames = [emotion_dir + '/' + emotion_frame for emotion_frame in os.listdir(emotion_dir)]
        emotion_frames = list(map(load_image, emotion_frames))
        print(emotion_frames)
        i = 0
        while True:
          frame = emotion_frames[i]
          cropped_img = cv2.resize(frame, (48, 48))
          cropped_img = np.array([cropped_img])
          cropped_img = cropped_img.astype('float32')
          cropped_img = cropped_img / 255;
          cropped_img = cropped_img[0:1]
          prediction = model.predict(cropped_img)
          maxindex = int(np.argmax(prediction))

          frame = cv2.resize(frame,(1600,960),interpolation = cv2.INTER_CUBIC);
          cv2.putText(frame, 'prediction : '+emotion_dict[maxindex], (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
          cv2.putText(frame, 'actual     : '+emotion, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
          cv2.imshow('Video', frame)
          #cap at 24 frame / second
          sleep(0.04)
          i = (i + 1) % len(emotion_frames)
          key = cv2.waitKey(1) & 0xFF
          if key == ord('q'):
            exit(0)

          if key == ord('n'):
            break
