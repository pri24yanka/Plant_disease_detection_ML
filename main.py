from email.mime import image
import streamlit as st
import tensorflow as tf
import numpy as np
import os
import gdown

# Google Drive file ID
file_id = "1baIMgeetYnJxmRJf3LyuYDt0mvbOMbAE"
model_path = "trained_plant_disease_model.keras"

# Function to download model from Google Drive
def download_model():
    if not os.path.exists(model_path):
        with st.spinner("Downloading model... Please wait."):
            gdown.download(f"https://drive.google.com/uc?export=download&id={file_id}", model_path, quiet=False)
            st.success("Model downloaded successfully!")

# TensorFlow Model Prediction
def model_prediction(test_image):
    download_model()  # Ensure model is downloaded before loading

    model = tf.keras.models.load_model(model_path)
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element


# Precautions for each disease class
precautions_dict = {
    'Apple___Apple_scab': [
        "Remove and destroy infected leaves from the ground. (Effective ~80%)",
        "Apply fungicides as a preventive measure. (Consult a professional for exact fungicide and timing; effective ~70%)",
        "Plant disease-resistant varieties if available. (Highly effective ~90%)"
    ],
    'Apple___Black_rot': [
        "Prune out all infected twigs and branches to prevent spread. (Effective ~85%)",
        "Ensure proper air circulation around the trees by spacing plants well. (Moderately effective ~75%)",
        "Apply fungicides during the growing season. (Consult a local expert for the best treatment; effective ~80%)"
    ],
    'Apple___Cedar_apple_rust': [
        "Remove nearby juniper plants, which can host the rust. (Highly effective ~90%)",
        "Apply fungicide to the apple trees during the growing season. (Effective ~80%)",
        "Plant resistant apple varieties if available. (Very reliable; check regional varieties for effectiveness)"
    ],
    'Blueberry___healthy': ["No action needed; the plant is healthy."],
    'Cherry_(including_sour)___Powdery_mildew': [
        "Apply fungicides to control powdery mildew. (Choose region-specific fungicides; ~75% effective)",
        "Prune infected areas to improve air circulation. (Effective ~70%)",
        "Avoid overhead watering to keep foliage dry. (Reliable ~80%)"
    ],
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': [
        "Rotate crops to prevent disease buildup in soil. (Effective in most cases ~85%)",
        "Use resistant hybrids if available. (Highly effective ~90%)",
        "Apply fungicides when symptoms first appear. (Effective, but follow local guidelines; ~75%)"
    ],
    'Corn_(maize)___Common_rust_': [
        "Plant resistant corn varieties. (Highly effective ~90%)",
        "Monitor fields regularly and apply fungicides if needed. (Check regional recommendations; ~70% effective)",
        "Avoid planting corn too densely to improve airflow. (Moderate effectiveness ~65%)"
    ],
    'Corn_(maize)___Northern_Leaf_Blight': [
        "Plant disease-resistant hybrids if available. (Effective ~85%)",
        "Rotate crops to reduce disease recurrence. (Very reliable ~80%)",
        "Apply fungicides during early stages of disease. (Follow professional advice for effectiveness; ~75%)"
    ],
    'Grape___Black_rot': [
        "Prune out infected vines and remove diseased fruit. (Effective ~85%)",
        "Train vines on trellises to ensure good air circulation. (Moderate reliability ~70%)",
        "Apply fungicides throughout the growing season. (Consult a local expert; effectiveness ~75%)"
    ],
    'Grape___Esca_(Black_Measles)': [
        "Prune out affected wood and remove diseased grapes. (Effective ~80%)",
        "Avoid wounding vines during pruning to prevent entry points for pathogens. (Reliable ~75%)",
        "Consider using fungicides if disease is severe. (Professional recommendation advised; ~65% effective)"
    ],
    'Orange___Haunglongbing_(Citrus_greening)': [
        "Remove and destroy infected trees to prevent spread. (Effective but costly; ~90%)",
        "Control psyllid populations, which spread the disease. (Moderate effectiveness; consult pest control experts)",
        "Use certified disease-free plants for new orchards. (Highly effective ~90%)"
    ],
    'Peach___Bacterial_spot': [
        "Apply copper-based bactericides as a preventive measure. (Effective ~75%)",
        "Avoid overhead irrigation to reduce leaf wetness. (Moderate reliability ~70%)",
        "Prune trees to improve air circulation. (Effective ~65%)"
    ],
    'Potato___Early_blight': [
        "Rotate crops to avoid disease buildup. (Highly effective ~85%)",
        "Apply fungicides as symptoms appear. (Follow professional advice for best results; ~75%)",
        "Remove and destroy infected plant debris. (Effective ~80%)"
    ],
    'Potato___Late_blight': [
        "Plant certified disease-free seed potatoes. (Highly effective ~90%)",
        "Apply fungicides regularly during wet weather. (Consult local guidelines; ~70% effective)",
        "Destroy infected tubers and plants to reduce spread. (Very effective ~85%)"
    ],
    'Tomato___Bacterial_spot': [
        "Use copper-based bactericides as a preventive measure. (Consult with local experts; ~70% effective)",
        "Avoid overhead watering to keep foliage dry. (Moderate reliability ~80%)",
        "Remove infected leaves to reduce spread. (Effective ~75%)"
    ],
    'Tomato___Early_blight': [
        "Apply fungicides at early signs. (Effective ~75%; follow professional guidance)",
        "Remove infected debris from the garden to limit spread. (Reliable ~80%)",
        "Rotate crops to avoid reintroducing disease in the soil. (Highly effective ~85%)"
    ],
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': [
        "Control whitefly populations, which spread the virus. (Moderate reliability ~70%)",
        "Remove and destroy infected plants. (Effective but labor-intensive; ~80%)",
        "Use resistant tomato varieties. (Highly effective ~85%)"
    ],
    'Tomato___healthy': ["No action needed; the plant is healthy."]
}


# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])


# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
                This dataset consists of about 87K RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets, preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purposes.
                #### Content
                1. Train (70,295 images)
                2. Test (33 images)
                3. Validation (17,572 images)
                """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    st.sidebar.subheader("NOTE")
    st.sidebar.markdown("""

    Effectiveness (%): Indicates the approximate success rate based on general research; 
    effectiveness may vary based on location and specific plant varieties.
                        
    Consult Experts: For many treatments, particularly fungicides and pesticides, it's best to consult local agricultural experts for precise recommendations and safe usage.
                        
    On Your Risk: While these precautions are generally effective, results may vary. """                    )
    
    test_image = st.file_uploader("Choose an Image:")
    if test_image is not None and st.button("Show Image"):
        st.image(image, use_container_width=True) 

    # Predict button
    if st.button("Predict"):
        st.write("Our Prediction")
        result_index = model_prediction(test_image)
        # Reading Labels
        class_name = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 
            'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
            'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 
            'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 
            'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
            'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 
            'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
            'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 
            'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 
            'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
            'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
            'Tomato___healthy'
        ]
        predicted_class = class_name[result_index]
        st.success(f"Model is predicting it's a {predicted_class}")

        # Show precautions for the predicted disease
        precautions = precautions_dict.get(predicted_class, ["No precautions available for this disease."])
        st.subheader("Precautions")
        for precaution in precautions:
            st.write(f"- {precaution}")
