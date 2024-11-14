import tensorflow as tf
import keras
from keras.models import Model,Sequential
from keras.layers import Dense, Activation, Flatten, Dropout, Lambda
from keras.activations import relu, softmax
from keras.layers import Input,LeakyReLU, MaxPooling2D, AveragePooling2D, Conv2DTranspose, Conv2D, Conv1D, BatchNormalization, UpSampling2D, Add, Concatenate, SeparableConv2D, GlobalAveragePooling2D, DepthwiseConv2D, Multiply, Reshape, Maximum, Minimum, Subtract

from keras.preprocessing.image import ImageDataGenerator
from keras.utils import array_to_img, img_to_array, load_img
from keras.initializers import TruncatedNormal
from keras import regularizers

from keras.optimizers import Adam, RMSprop, SGD

from keras import initializers

from keras.models import load_model,save_model

from keras.callbacks import ModelCheckpoint,EarlyStopping,LearningRateScheduler,TerminateOnNaN,ReduceLROnPlateau, TensorBoard

from keras import backend as K

from keras.applications.mobilenet_v2 import MobileNetV2,preprocess_input
from keras.applications.mobilenet import MobileNet
from keras.applications.resnet import ResNet50

from tensorflow import keras
from keras import layers
from keras.callbacks import LearningRateScheduler,ModelCheckpoint
from keras.losses import CategoricalCrossentropy


import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import math
import cv2

from sklearn.metrics import classification_report, confusion_matrix

from model import make_ACFF_model

val_data_dir=  r"C:\Users\kvjai\ML PROJECTS\AIDER EmergencyNet accident detection\data\AIDER"

seed = 22
rnd.seed(seed)
np.random.seed(seed)

dsplit = 0.2

img_height=224
img_width=224
W=img_width
num_classes = 5
num_workers=1
batch_size=128
epochs = 50
lr_init=1e-1

validation_datagen = ImageDataGenerator(rescale=1./255.,
    preprocessing_function = None,validation_split=dsplit)

validation_generator = validation_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
    )


opt = tf.keras.optimizers.SGD(lr=1e-2,momentum=0.9)
loss = CategoricalCrossentropy()

#model = load_model('../results/model.h5')
inp,cls = make_ACFF_model(img_height,img_width,C=num_classes)
model = Model(inputs=[inp], outputs=[cls])
model.load_weights('../results/model_weights.h5')
model.summary()
model.compile(optimizer=opt,metrics=keras.metrics.CategoricalAccuracy(),loss=loss)

score = model.evaluate(validation_generator)
print(score)

#print(cor,cor/validation_generator.samples)


# Generate predictions
Y_pred = model.predict(validation_generator, steps=validation_generator.samples, batch_size=1)
y_pred = np.argmax(Y_pred, axis=1)

# Retrieve class labels from validation generator
class_labels = list(validation_generator.class_indices.keys())

# Convert predicted indices to class labels
predicted_classes = [class_labels[i] for i in y_pred]

# Print the predicted class values for each sample
print("Predicted class values:")
for i, label in enumerate(predicted_classes):
    print(f"Sample {i + 1}: Predicted Class = {label}")

# Optionally, print the confusion matrix and classification report as well
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))

print('Classification Report')
print(classification_report(validation_generator.classes, y_pred, target_names=class_labels))





# Function to predict class for a single image
def predict_image_class(image_path):
    # Load the image and resize it to the required input size
    img = load_img(image_path, target_size=(img_height, img_width))
    img_array = img_to_array(img) / 255.0  # Rescale to match training scaling
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict the class
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction, axis=1)[0]
    predicted_class = class_labels[predicted_index]

    print(f"Predicted class for the image is: {predicted_class}")

# Example usage: prompt the user for an image file path and predict its class
image_path = r"C:\Users\kvjai\ML PROJECTS\Road debris detection\data\test\153_jpeg.rf.e54033080b414081442600b95e796221.jpg"
predict_image_class(image_path)


