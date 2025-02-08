import gradio as gr
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt

# Load the trained model
model = tf.keras.models.load_model("tomato_disease_model.h5")

# Class labels (Replace with actual class names from your dataset)
class_names = [
    "Bacterial Spot",
    "Early Blight", 
    "Healthy",
    "Late Blight",
    "Leaf Mold",
    "Septoria Leaf Spot",
    "Spider Mites",  
    "Target Spot",
    "Mosaic Virus",
    "Yellow Leaf Curl Virus",
]

# Preprocessing function
def preprocess_image(image):
    img = image.convert("RGB")  # Convert to RGB (PIL format)
    img = img.resize((299, 299))  # Resize to InceptionV3 input size
    img = np.array(img) / 255.0  # Normalize pixel values (0 to 1)
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Prediction function
def predict_disease(image):
    processed_image = preprocess_image(image)
    predictions = model.predict(processed_image)
    
    class_index = np.argmax(predictions)  # Get class with highest probability
    confidence = np.max(predictions)  # Get confidence score

    # Generate confidence bar chart
    fig, ax = plt.subplots()
    ax.barh(class_names, predictions[0], color="skyblue")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_title("Prediction Confidence Scores")

    return f"Prediction: {class_names[class_index]} (Confidence: {confidence:.2f})", fig

# Create the Gradio interface
interface = gr.Interface(
    fn=predict_disease,
    inputs=gr.Image(type="pil"),  # Accepts PIL image
    outputs=["text", gr.Plot()],  # Returns prediction text + confidence plot
    title="🍅 Tomato Disease Detection",
    description="Upload an image of a tomato leaf to detect disease. The model will predict the type of disease with confidence scores.",
    theme="default",  # Use default theme
    allow_flagging="manual"  # Users can flag incorrect results
)

# Launch the app
interface.launch()
