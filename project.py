import streamlit as st
import tensorflow as tf
import numpy as np

from PIL import Image
import io

# TensorFlow model prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model('trained_model.keras')
    
    # Convert the file-like object to an image
    image = Image.open(test_image).convert('RGB')
    image = image.resize((128, 128))
    
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    prediction = model.predict(input_arr)
    result_index = np.argmax(prediction)
    return result_index

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox(
    "Choose an option",  # Label for the selectbox
    ["Home", "About", "Disease Recognition"]  # List of options
)

# Home page
if app_mode == 'Home':
    st.header("PlantDoctor: Recognise Your Plant\'s Disease")
    image_path = "homeImg.jpg"
    st.image(image_path, use_column_width=True)
    st.markdown("""**Harness the Power of AI to Keep Your Plants Thriving!**""")

# About page
elif app_mode == "About":
    st.header("About")
    st.markdown("""**Is your green thumb feeling a little rusty?**
                Don't worry, you're not alone! Plant diseases can be tricky to spot, even for experienced gardeners. That's where we come in.

Our app is designed to help you identify potential problems with your plants quickly and easily. Using advanced technology, we've created a model that have 38 classes for plants and their disease. It's like having a plant doctor right in your pocket!

With an impressive accuracy rate of 92%, you can trust our app to give you reliable results. So next time you notice something amiss with your beloved greenery, simply snap a picture and let our app do the diagnosing.

**Let's keep your plants happy and healthy together!**
""")

# Prediction page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an image: ", type=['jpg', 'png', 'jpeg'])
    
    if test_image is not None:
        st.image(test_image, use_column_width=True)
        
        # Predict button
        if st.button("Predict"):
            with st.spinner("Predicting..."):
                result_index = model_prediction(test_image)

                # Define class names
                class_name = ['Apple___Apple_scab',
 'Apple___Black_rot',
 'Apple___Cedar_apple_rust',
 'Apple___healthy',
 'Blueberry___healthy',
 'Cherry_(including_sour)___Powdery_mildew',
 'Cherry_(including_sour)___healthy',
 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
 'Corn_(maize)___Common_rust_',
 'Corn_(maize)___Northern_Leaf_Blight',
 'Corn_(maize)___healthy',
 'Grape___Black_rot',
 'Grape___Esca_(Black_Measles)',
 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
 'Grape___healthy',
 'Orange___Haunglongbing_(Citrus_greening)',
 'Peach___Bacterial_spot',
 'Peach___healthy',
 'Pepper,_bell___Bacterial_spot',
 'Pepper,_bell___healthy',
 'Potato___Early_blight',
 'Potato___Late_blight',
 'Potato___healthy',
 'Raspberry___healthy',
 'Soybean___healthy',
 'Squash___Powdery_mildew',
 'Strawberry___Leaf_scorch',
 'Strawberry___healthy',
 'Tomato___Bacterial_spot',
 'Tomato___Early_blight',
 'Tomato___Late_blight',
 'Tomato___Leaf_Mold',
 'Tomato___Septoria_leaf_spot',
 'Tomato___Spider_mites Two-spotted_spider_mite',
 'Tomato___Target_Spot',
 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
 'Tomato___Tomato_mosaic_virus',
 'Tomato___healthy']
                
                st.success(f"Model is predicting it is a {class_name[result_index]}")
