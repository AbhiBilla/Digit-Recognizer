#Importing required packages
! pip install streamlit --quiet
! pip install pyngrok==4.1.1 --quiet
! pip install streamlit-drawable-canvas --quiet
from pyngrok import ngrok

#Web deployment code
%%writefile app.py
import streamlit as st
import tensorflow as tf
import cv2 #ComputerVision
from google.colab.patches import cv2_imshow
import numpy as np
model = tf.keras.models.load_model("Minor_Project2.hdf5")
from streamlit_drawable_canvas import st_canvas
st.title("Digit-Recognizer")
# Create a canvas component
canvas_result = st_canvas(
    # Fixed fill color with some opacity
    stroke_width=10,
    stroke_color=' #ffffff',
    background_color='#000000',
    height=200,width=200,
    drawing_mode="freedraw")

# Do something interesting with the image data and paths
if canvas_result.image_data is not None:
  r_img = cv2.resize(canvas_result.image_data.astype('uint8'),(28,28))
  x_test = cv2.cvtColor(r_img,cv2.COLOR_BGR2GRAY)
  y_pred = model.predict(x_test.reshape(1,28,28))
  st.write(f'result: {np.argmax(y_pred[0])}')


#Getting URL
! nohup streamlit run app.py &
url = ngrok.connect(port='8501')
url