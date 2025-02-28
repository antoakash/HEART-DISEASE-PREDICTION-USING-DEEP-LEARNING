import gradio as gr
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Path to the saved model
model_path = '/Users/akash/Desktop/MRI DATASET PREDICTION/heart_disease_mri_densenet_model.h5'

# Load the trained model
model = load_model(model_path)

# Categories for prediction
categories = ["heart_failure_with_infarct", "heart_failure_without_infarct", "hypertrophy", "normal"]

# Preprocessing the new image for prediction
def preprocess_image(image_path, target_size=(224, 224)):
    # Load the image
    img = load_img(image_path, target_size=target_size)
    # Convert the image to an array
    img_array = img_to_array(img)
    # Expand dimensions to match the input format of the model
    img_array = np.expand_dims(img_array, axis=0)
    # Preprocess the image (rescale as done in training)
    img_array = img_array / 255.0
    return img, img_array

# Prediction function
def predict_condition(image):
    # Preprocess the image
    img, img_array = preprocess_image(image)
    
    # Make predictions
    predictions = model.predict(img_array)
    
    # Decode the prediction
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class = categories[predicted_class_index]
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.imshow(img)
    
    # Annotate the prediction on the image
    ax.add_patch(patches.Rectangle((0, 0), img.size[0], img.size[1], linewidth=2, edgecolor='red', facecolor='none'))
    ax.text(10, 20, f"Predicted: {predicted_class}", color='white', fontsize=14, backgroundcolor='black')

    ax.axis('off')  # Hide axes
    
    # Return the plot and the prediction message
    return fig, f"The predicted condition is: {predicted_class}"

# Gradio interface
interface = gr.Interface(
    fn=predict_condition,
    inputs=gr.Image(type="filepath", label="Upload MRI Image"),  # Change type to 'filepath'
    outputs=[gr.Plot(label="Image with Prediction"), gr.Textbox(label="Prediction")],
    title="Heart Disease Detection using DenseNet",
    description="Upload an MRI image to predict the heart disease condition."
)

interface.launch(share=True)
