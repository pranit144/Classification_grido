import os
import cv2
import numpy as np
import gradio as gr
from cvzone.ClassificationModule import Classifier

# Load the trained model and labels
classifier = Classifier('keras_model.h5', 'labels.txt')

# Classification dictionary for bin mapping
classDic = {
    0: "Nothing",
    1: "Zip-top cans",
    2: "Newspaper",
    3: "Apple",
    4: "Watercolor pen",
    5: "Disinfectant",
    6: "Battery",
    7: "Vegetable leaf",
    8: "Old shoes"
}


# Classification function
def classify_waste(image):
    """ Classify the waste based on the uploaded image. """
    img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # Convert RGB (Gradio) to BGR (OpenCV)
    prediction = classifier.getPrediction(img)
    class_id = prediction[1]
    
    waste_category = classDic.get(class_id, "Unknown")
    return f"Predicted Category: {waste_category}"

# Create Gradio Interface
iface = gr.Interface(
    fn=classify_waste,
    inputs=gr.Image(type="numpy"),
    outputs="text",
    title="Classification Model",
    description="Upload an image, and the model will classify it into the appropriate category."
)

# Launch the app
iface.launch()
