import cv2
import PIL
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers,models
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import Callback, EarlyStopping,ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import Model
from tensorflow.keras.layers.experimental import preprocessing



#code for opening camera and taking picture
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
img_counter = 0
while True:
    ret, frame = cap.read()
    cv2.imshow('frame',frame)
    if not ret:
        break
    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    if k%256 == 32:
        # SPACE pressed
        img_name = "opencv_frame_{}.png".format(img_counter)
        cv2.imwrite(img_name, frame)
        print("{} written!".format(img_name))
        break
cap.release()
cv2.destroyAllWindows()

#code for resizing image to (224, 224) from (640, 480)
img = PIL.Image.open('opencv_frame_0.png')
img = img.resize((224, 224))
img.save('opencv_frame_0.png')


#code for loading model and capturing the data from camera and predicting the image
# Load the model
model = tf.keras.models.load_model('garbage_classifier_new.h5')
#model.summary()
# Load the saved image using keras and resize it to (224, 224)
img = keras.preprocessing.image.load_img(
    "opencv_frame_0.png", target_size=(224, 224)
)
# Convert the image to array
img_array = keras.preprocessing.image.img_to_array(img)
# Expand the dimensions of the image
img_array = tf.expand_dims(img_array, 0)
# Predict the image
predictions = model.predict(img_array)

# Get the predicted class
score = tf.nn.softmax(predictions[0])
class_names = ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']

if __name__ == '__main__':
    print(
    "This image most likely belongs to {} with a {:.2f} percent confidence."
    .format(class_names[np.argmax(score)], 100 * np.max(score))
    )

#code for displaying the imagewith annotation of the predicted class
img = cv2.imread('opencv_frame_0.png')
img = cv2.putText(img, class_names[np.argmax(score)], (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
cv2.imshow('image', img)
cv2.waitKey(2000)
cv2.destroyAllWindows()





# arduino connection code
def get_value():
    return class_names[np.argmax(score)];
    