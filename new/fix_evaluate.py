import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os
import csv

# Paths and parameters
model_path = 'C://Users//Sinta//Documents//Tugas Akhir//Backup Plan//sign_language_detection//scripts//new//models//sign_language_cnn_model_best.h5'
class_labels_path = 'C://Users//Sinta//Documents//Tugas Akhir//Backup Plan//sign_language_detection//scripts//new//class_labels_model_best.json'
test_dir = 'C://Users//Sinta//Documents//Tugas Akhir//Backup Plan//sign_language_detection//dataset//test'
csv_path = 'C://Users//Sinta//Documents//Tugas Akhir//Backup Plan//sign_language_detection//scripts//new//evaluation_results.csv'

batch_size = 32
img_height = 64
img_width = 64

# Load the model
model = tf.keras.models.load_model(model_path)

# Load class labels
with open(class_labels_path, 'r') as f:
    class_labels = json.load(f)

# Prepare data generator for the test set
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False)

# Evaluate the model
scores = model.evaluate(test_generator)
accuracy = scores[1]

print(f'Evaluation completed. Accuracy: {accuracy}')
