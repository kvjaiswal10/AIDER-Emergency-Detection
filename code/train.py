import tensorflow as tf
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import array_to_img, img_to_array, load_img
from keras.optimizers import Adam, RMSprop, SGD
from keras.models import load_model, save_model
from keras.models import Model
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras import backend as K
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from keras.losses import CategoricalCrossentropy

import cv2
import random as rnd
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.metrics import classification_report, confusion_matrix

# External imports from custom modules
from ImageDataAugmentor.image_data_augmentor import ImageDataAugmentor
from augment import create_augmentations
from model import make_ACFF_model
from other import *

K.clear_session()

# Directory and parameters
train_data_dir = r"C:\Users\kvjai\ML PROJECTS\AIDER EmergencyNet accident detection\data\AIDER"
val_data_dir = r"C:\Users\kvjai\ML PROJECTS\AIDER EmergencyNet accident detection\data\AIDER"

img_height = 240
img_width = 240
num_classes = 5
num_workers = 1
batch_size = 64
epochs = 80
lr_init = 1e-2

# Build model
inp, cls = make_ACFF_model(img_height, img_width, C=num_classes)
model = Model(inputs=[inp], outputs=[cls])
model.summary()

# Data augmentation
AUGMENTATIONS = create_augmentations(img_height, img_width, p=0.1)
seed = 22
rnd.seed(seed)
np.random.seed(seed)
dsplit = 0.2

train_datagen = ImageDataAugmentor(
    rescale=1. / 255.,
    augment=AUGMENTATIONS,
    validation_split=dsplit
)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

validation_datagen = ImageDataGenerator(rescale=1. / 255., validation_split=dsplit)

validation_generator = validation_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Checkpoints and callbacks
checkpoint = ModelCheckpoint('../results/model.h5', monitor='val_categorical_accuracy', save_best_only=True, mode='max', verbose=1, save_weights_only=False)
weight_checkpoint = ModelCheckpoint('../results/model_weights.h5', monitor='val_categorical_accuracy', save_best_only=True, mode='max', verbose=1, save_weights_only=True)
opt = tf.keras.optimizers.SGD(lr=lr_init, momentum=0.9)

# Learning rate scheduling
def cosine_decay(epoch, initial_lrate=lr_init, epochs_tot=epochs, period=1, fade_factor=1.0, min_lr=1e-3):
    return max(min_lr, 0.5 * initial_lrate * (1 + math.cos(epoch * math.pi / period)))

lrs = LearningRateScheduler(cosine_decay, verbose=1)
lrr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6, verbose=1)

# Early stopping
early_stopping = EarlyStopping(
    monitor="val_categorical_accuracy",
    patience=10,  # stops after 10 epochs with no significant improvement
    min_delta=0.0005,  # minimum improvement to qualify as an increase
    mode="max",
    verbose=1
)

# Add all callbacks to list
callbacks_list = [lrs, checkpoint, weight_checkpoint, early_stopping]

# Compile model
SMOOTHING = 0.1
loss = CategoricalCrossentropy(label_smoothing=SMOOTHING)
model.compile(optimizer=opt, metrics=[keras.metrics.CategoricalAccuracy()], loss=loss)

# Train the model
history = model.fit(
    x=train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    callbacks=callbacks_list,
    workers=num_workers,
    class_weight={0: 1.0, 1: 1.0, 2: 1.0, 3: 0.35, 4: 1.0}
)

# Plotting accuracy and loss
plt.plot(history.history['categorical_accuracy'])
plt.plot(history.history['val_categorical_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('../results/acc.png')
plt.clf()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.savefig('../results/loss.png')

# Load best model and evaluate
model = load_model('../results/model.h5')
score = model.evaluate(validation_generator)
print("Evaluation Score:", score)

# Generate confusion matrix and classification report
Y_pred = model.predict(validation_generator, steps=validation_generator.samples, batch_size=1)
y_pred = np.argmax(Y_pred, axis=1)
print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))

target_names = ['collapsed_building', 'fire', 'flooded_areas', 'normal', 'traffic_incident']
print('Classification Report')
print(classification_report(validation_generator.classes, y_pred, target_names=target_names))
