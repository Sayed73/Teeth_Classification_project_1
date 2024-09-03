import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model


model = load_model('mobileNet_model.h5')


class_names = ['CaS', 'CoS', 'Gum', 'MC', 'OC', 'OLP', 'OT']

st.title('Teeth Disease Classification App')


upload_file = st.file_uploader("Choose an Image..", type ="jpg")
if upload_file is not None:
    image = Image.open(upload_file)
    st.image(image, caption = "Uploaded Image.", use_column_width = True)
    
    
   # preprocess the uploaded image 
    image = image.resize((224,224))
    image_array = tf.keras.preprocessing.image.img_to_array(image)
    image_array = image_array/225.0
    image_array = np.expand_dims(image_array, axis=0)
    
    # predict the class for the uploaded image 
    
    prediction = model.predict(image_array)
    predicted_class = class_names[np.argmax(prediction)]
    conf = np.max(prediction)
    
    st.write(f"Prediction: {predicted_class}")
    st.write(f"Confidance: {conf:.2f}")
    
    st.bar_chart(prediction[0])
    
st.write("Upload an image to get a prediction.")
