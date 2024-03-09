!pip install skimage
import numpy as np
import os
from skimage import io
from skimage.transform import resize
from sklearn.feature_extraction import image
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.fftpack import dct
import tensorflow as tf

# Function to extract DCT features from an image
def extract_dct(img):
    img = dct(img, type=2, norm="ortho", axis=0)
    img = dct(img, type=2, norm="ortho", axis=1)
    img = np.abs(img)
    img+= 1e-13
    img = tf.math.log(img)
    img-=np.mean(img)
    img=np.divide(img,np.std(img))
    return img


# Function to load images from folder
def load_images_from_folder(folder_path, image_size=(128, 128)):
    images = []
    for filename in os.listdir(folder_path):
        img = io.imread(os.path.join(folder_path, filename), as_gray=True)
        img = resize(img, image_size)
        images.append(img)
    return images

# Load real and fake images from folders
real_images = load_images_from_folder('/content/drive/MyDrive/Fake images/Dataset/Real Images')
fake_images = load_images_from_folder('/content/drive/MyDrive/Fake images/Dataset/Fake Images')

# Extract DCT features from images
X_real = np.array([extract_dct(img) for img in real_images])
X_fake = np.array([extract_dct(img) for img in fake_images])

# Create labels for real and fake images
y_real = np.ones(X_real.shape[0])
y_fake = np.zeros(X_fake.shape[0])

# Combine real and fake images and labels
X = np.vstack((X_real, X_fake))
y = np.hstack((y_real, y_fake))

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Ridge Regression classifier
ridge_clf = RidgeClassifier(alpha=1.0)
ridge_clf.fit(X_train, y_train)

# Predict on the testing set
y_pred = ridge_clf.predict(X_test)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

from google.colab import drive
drive.mount('/content/drive')

import numpy as np
import os
from skimage import io
from skimage.transform import resize
from sklearn.feature_extraction import image
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.fftpack import dct
import tensorflow as tf
import matplotlib.pyplot as plt
import gradio as gr
# Function to extract DCT features from an image
def extract_dct(img):
    img = dct(img, type=2, norm="ortho", axis=0)
    img = dct(img, type=2, norm="ortho", axis=1)
    img = np.abs(img)
    img += 1e-13
    img = np.log(img)
    img -= np.mean(img)
    img /= np.std(img)

    return img.flatten()

# Function to load images from folder
def load_images_from_folder(folder_path, image_size=(128, 128)):
    images = []
    for filename in os.listdir(folder_path):
        img = io.imread(os.path.join(folder_path, filename), as_gray=True)
        img = resize(img, image_size)
        images.append(img)
    return images

# Load real and fake images from folders
real_images = load_images_from_folder('/content/drive/MyDrive/Fake images/Dataset/Real Images')
fake_images = load_images_from_folder('/content/drive/MyDrive/Fake images/Dataset/Fake Images')

# Extract DCT features from images
X_real = np.array([extract_dct(img) for img in real_images])
X_fake = np.array([extract_dct(img) for img in fake_images])
# Create labels for real and fake images
y_real = np.ones(len(X_real))
y_fake = np.zeros(len(X_fake))
# Combine real and fake images
X = np.vstack((X_real, X_fake))
# Combine real and fake labels
y = np.hstack((y_real, y_fake))
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train Ridge Regression classifier
ridge_clf = RidgeClassifier(alpha=1.0)
ridge_clf.fit(X_train, y_train)
# Predict on the testing set
y_pred = ridge_clf.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
import pickle
with open('my_model.pkl', 'wb') as f:
    pickle.dump(ridge_clf, f)

import numpy as np
import pickle
from skimage import io
from skimage.transform import resize
from scipy.fftpack import dct

def extract_dct(img):
    img = dct(img, type=2, norm="ortho", axis=0)
    img = dct(img, type=2, norm="ortho", axis=1)
    img = np.abs(img)
    img += 1e-13
    img = np.log(img)
    img -= np.mean(img)
    img /= np.std(img)
    return img.flatten()

def preprocess_image(image_path, image_size=(128, 128)):
    img = io.imread(image_path, as_gray=True)
    img = resize(img, image_size)
    dct_features = extract_dct(img)
    return dct_features

# Load the saved model
with open('my_model.pkl', 'rb') as f:
    ridge_clf = pickle.load(f)

# Get the file path of the input image and preprocess it
image_path = "/content/00003.png"
dct_features = preprocess_image(image_path)

# Make a prediction on the preprocessed image
prediction = ridge_clf.predict([dct_features])
if prediction[0] == 1:
    label = "Real"
elif prediction[0] == 0:
    label = "Fake"
else:
    print("Error")
print(f"Prediction: {label}")

import streamlit as st
import numpy as np
import pickle
from skimage import io
from skimage.transform import resize
from scipy.fftpack import dct

# Load the pre-trained model
with open('my_model.pkl', 'rb') as f:
    ridge_clf = pickle.load(f)

# Function to preprocess the image
def preprocess_image(image_path, image_size=(128, 128)):
    img = io.imread(image_path, as_gray=True)
    img = resize(img, image_size)
    img = extract_dct(img)
    return img

# Function to extract DCT features
def extract_dct(img):
    img = dct(img, type=2, norm="ortho", axis=0)
    img = dct(img, type=2, norm="ortho", axis=1)
    img = np.abs(img)
    img += 1e-13
    img = np.log(img)
    img -= np.mean(img)
    img /= np.std(img)
    return img.flatten()

# Function to make prediction
def predict_image(image):
    try:
        # Preprocess the image
        dct_features = preprocess_image(image)
        # Make prediction
        prediction = ridge_clf.predict([dct_features])
        if prediction[0] == 1:
            label = "Real"
        elif prediction[0] == 0:
            label = "Fake"
        else:
            label = "Unknown"
        return label
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        return "Error"

import numpy as np
import os
from skimage import io
from skimage.transform import resize
from sklearn.feature_extraction import image
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from scipy.fftpack import dct
import tensorflow as tf
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
import tensorflow as tf
from tensorflow.python.framework import ops


np.random.seed(1)
# Function to extract DCT features from an image
def extract_dct(img):
    img = dct(img, type=2, norm="ortho", axis=0)
    img = dct(img, type=2, norm="ortho", axis=1)
    img = np.abs(img)
    img += 1e-13
    img = np.log(img)
    img -= np.mean(img)
    img /= np.std(img)
    return img.flatten()

def pixel_image(img):
   return img.flatten()

# Function to load images from folder
def load_images_from_folder(folder_path, image_size=(128, 128)):
    images = []
    for filename in os.listdir(folder_path):
        img = io.imread(os.path.join(folder_path, filename), as_gray=True)
        img = resize(img, image_size)
        images.append(img)
    return images

# Load real and fake images from folders
real_images = load_images_from_folder('/content/drive/MyDrive/Fake images/Dataset/Real Images')
fake_images = load_images_from_folder('/content/drive/MyDrive/Fake images/Dataset/Fake Images')
index = 124
plt.imshow(fake_images[index]) #display sample training image
plt.show()
# Extract DCT features from images
X_real = np.array([pixel_image(img) for img in real_images])
X_fake = np.array([pixel_image(img) for img in fake_images])
# Create labels for real and fake images
y_real = np.ones(len(X_real))
y_fake = np.zeros(len(X_fake))
# Combine real and fake images
X = np.vstack((X_real, X_fake))
# Combine real and fake labels
y = np.hstack((y_real, y_fake))
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train Ridge Regression classifier
ridge_clf = RidgeClassifier(alpha=1.0)
ridge_clf.fit(X_train, y_train)
# Predict on the testing set
y_pred = ridge_clf.predict(X_test)
# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
