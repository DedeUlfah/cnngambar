import json
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_dir = r"C:\Users\LABKOM2\Downloads\data\data30"
test_dir = r"C:\Users\LABKOM2\Downloads\data\data30 test"

batch_size = 32
img_height = 224
img_width = 224
# Load the best model
model = load_model(r'C:\Users\LABKOM2\cnngambar\new\models1\sign_language_cnn_resnet50_model.h5')

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

num_classes = len(train_generator.class_indices)
# Evaluasi model pada test set
test_loss, test_accuracy = model.evaluate(test_generator)

# Menyimpan hasil evaluasi ke file JSON
evaluation_results = {
    'test_loss': test_loss,
    'test_accuracy': test_accuracy
}

# Save evaluation results to a JSON file
evaluation_file = 'evaluation_results.json'
with open(evaluation_file, 'w') as f:
    json.dump(evaluation_results, f)

print(f"Model evaluation results saved to {evaluation_file}")
print(f"Test Loss: {test_loss}")
print(f"Test Accuracy: {test_accuracy}")
