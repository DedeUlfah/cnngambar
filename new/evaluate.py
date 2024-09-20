import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
import csv
import json
import os

test_dir = 'C://Users//Sinta//Documents//Tugas Akhir//Backup Plan//sign_language_detection//dataset//test'

batch_size = 32
img_height = 64
img_width = 64

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=False
)

# Define the combinations used in training
combinations = [
    ((3, 3), 'relu', 'Adam'),
    ((3, 3), 'relu', 'SGD'),
    ((3, 3), 'relu', 'RMSprop'),
    ((3, 3), 'tanh', 'Adam'),
    ((3, 3), 'tanh', 'SGD'),
    ((3, 3), 'tanh', 'RMSprop'),
    ((3, 3), 'sigmoid', 'Adam'),
    ((3, 3), 'sigmoid', 'SGD'),
    ((3, 3), 'sigmoid', 'RMSprop'),
    ((5, 5), 'relu', 'Adam'),
    ((5, 5), 'relu', 'SGD'),
    ((5, 5), 'relu', 'RMSprop'),
    ((5, 5), 'tanh', 'Adam'),
    ((5, 5), 'tanh', 'SGD'),
    ((5, 5), 'tanh', 'RMSprop')
]

# Path to the models directory
models_dir = 'C://Users//Sinta//Documents//Tugas Akhir//Backup Plan//sign_language_detection//models'

# File to save the evaluation results
csv_file = 'C://Users//Sinta//Documents//Tugas Akhir//Backup Plan//sign_language_detection//models//evaluation_results.csv'

# Create the CSV file and write the header
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Model Index', 'Kernel Size', 'Activation', 'Optimizer', 'Training Accuracy', 'Test Accuracy'])

# Evaluate each model and save the results
for idx, (kernel_size, activation, optimizer) in enumerate(combinations):
    model_path = os.path.join(models_dir, f'sign_language_cnn_model_{idx+1}.h5')
    history_path = os.path.join(models_dir, f'history_model_{idx+1}.json')
    
    if os.path.exists(model_path) and os.path.exists(history_path):
        print(f"Evaluating model {idx+1}/{len(combinations)}: {model_path}")
        
        # Load the model
        model = load_model(model_path)
        
        # Load the training history
        with open(history_path, 'r') as f:
            history = json.load(f)
        
        # Get the last training accuracy
        training_accuracy = history['accuracy'][-1]
        
        # Evaluate the model on the test set
        loss, test_accuracy = model.evaluate(test_generator, verbose=1)
        
        # Append the results to the CSV file
        with open(csv_file, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([idx+1, kernel_size, activation, optimizer, training_accuracy, test_accuracy])
    else:
        print(f"Model or history not found for model index {idx+1}.")
