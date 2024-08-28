import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from keras.utils import to_categorical
import os

# Load the saved model
model = load_model('cnn_teeth_classification_model.h5')

# Initialize LabelEncoder
label_encoder = LabelEncoder()

def load_and_preprocess_image(uploaded_file, image_size=(224, 224)):
    """
    Function to load and preprocess a single image file.
    """
    # Read the image
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    # Resize the image
    img = cv2.resize(img, image_size)
    # Convert the image from BGR to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Normalize the image data
    img = img.astype('float32') / 255.0
    return img

def predict_disease(img):
    img_array = np.expand_dims(img, axis=0)
    prediction = model.predict(img_array)
    predicted_class_index = np.argmax(prediction)
    confidence = np.max(prediction)
    return predicted_class_index, confidence

# Streamlit app
st.title('Teeth Disease Classification')

# Load and fit the LabelEncoder with your class names
class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC' ,'OLP','OT']
label_encoder.fit(class_names)

# Show class names and their encoded values
# st.write("Class Names and Encoded Values:")
# for i, class_name in enumerate(class_names):
#     st.write(f"{class_name}: {label_encoder.transform([class_name])[0]}")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    img = load_and_preprocess_image(uploaded_file)
    st.image(img, caption='Uploaded Image', use_column_width=False, width=256)

    if st.button('Predict Disease'):
        predicted_class_index, confidence = predict_disease(img)
        predicted_class = label_encoder.inverse_transform([predicted_class_index])[0]
        st.markdown(f"### Predicted Disease: **{predicted_class}**")
        # st.write(f"Confidence: {confidence:.2f}")
