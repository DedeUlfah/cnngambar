import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import pandas as pd

# Load the trained model
model_path = 'C://Users//ASUS//Documents//tugasakhir//new//model30//sign_language_cnn_model_1.h5'
model = load_model(model_path)

# Load class labels
class_labels_path = 'C://Users//ASUS//Documents//tugasakhir//new//model30//class_labels_model_1.json'
with open(class_labels_path, 'r') as f:
    class_labels = json.load(f)
class_labels = {int(k): v for k, v in class_labels.items()}

# Define the path to the folder containing the test images
test_dir = 'C://Users//ASUS//Documents//tugasakhir//data_makanan-20240917T140626Z-001//test'
img_height = 64
img_width = 64

# Function to load and preprocess a single image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Create batch axis
    img_array /= 255.0  # Rescale
    return img_array

# Define the desired size for the images in the plot
desired_size = (380, 180)  # Set this to the size you want for the images

# List to store predictions
predictions = []

# Create a figure for the plots
fig, axes = plt.subplots(10, 5, figsize=(16, 16), dpi=80)
axes = axes.flatten()  # Flatten to simplify indexing
image_index = 0  # Index to keep track of image positions in the plot

# Loop through each sub-folder in the test directory
for class_folder in os.listdir(test_dir):
    class_folder_path = os.path.join(test_dir, class_folder)
    if not os.path.isdir(class_folder_path):
        continue

    # Loop through the images in each sub-folder
    for img_name in os.listdir(class_folder_path):
        img_path = os.path.join(class_folder_path, img_name)
        img_array = load_and_preprocess_image(img_path)

        # Make a prediction
        preds = model.predict(img_array)
        predicted_class = np.argmax(preds, axis=1)

        # Get the predicted label and accuracy
        predicted_label = class_labels[predicted_class[0]]
        predicted_accuracy = np.max(preds) * 100  # Convert to percentage

        # Append prediction to list
        predictions.append({
            'Image': img_name,
            'True Label': class_folder,
            'Predicted Label': predicted_label,
            'Accuracy (%)': predicted_accuracy
        })

        # Load the image for plotting
        img = PILImage.open(img_path)
        img = img.resize(desired_size)  # Resize the image
        img = np.array(img) / 255.0  # Convert to array and normalize

        # Plot the image and its prediction
        if image_index < len(axes):
            axes[image_index].imshow(img)
            axes[image_index].axis('off')
            # Add text below the image
            axes[image_index].text(0.5, -0.2, f"Predicted: {predicted_label}\nAccuracy: {predicted_accuracy:.2f}%", 
                                   ha='center', va='top', fontsize=8, transform=axes[image_index].transAxes)
            image_index += 1

# Hide any unused subplots
for j in range(image_index, len(axes)):
    axes[j].axis('off')

# Adjust the layout to make room for the titles
plt.subplots_adjust(top=0.95, bottom=0.05, left=0.05, right=0.95, hspace=0.6, wspace=0.3)

plt.show()

# Save predictions to CSV
predictions_df = pd.DataFrame(predictions)
csv_path = 'C://Users//ASUS//Documents//tugasakhir//new//model30//predictions.csv'
predictions_df.to_csv(csv_path, index=False)

print(f"Predictions saved to {csv_path}")
