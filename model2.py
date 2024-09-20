import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

# Path ke folder dataset gambar
folder_path = r'C:\Users\ASUS\Downloads\data_makanan-20240917T140626Z-001\data_makanan'

# Preprocessing data, scaling, dan splitting dataset (80% train, 20% validation)
datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)

# Generator untuk training data
train_generator = datagen.flow_from_directory(
    folder_path,
    target_size=(224, 224),  # Ukuran input untuk ResNet50
    batch_size=16,
    class_mode='categorical',
    subset='training')  # Untuk data training

# Generator untuk validation data
valid_generator = datagen.flow_from_directory(
    folder_path,
    target_size=(224, 224),
    batch_size=16,
    class_mode='categorical',
    subset='validation')  # Untuk data validation

# Load ResNet50 pre-trained model tanpa lapisan fully connected terakhir (include_top=False)
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze semua layer dari ResNet50
for layer in base_model.layers:
    layer.trainable = False

# Bangun model Sequential
model = Sequential([
    base_model,
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(24, activation='softmax')  # Sesuaikan dengan jumlah kelas (misal 24 kelas)
])

# Compile model
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Train the model
history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=30,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    validation_steps=valid_generator.samples // valid_generator.batch_size,
    callbacks=[early_stopping],
    verbose=1
)

# Prediksi menggunakan validation data
val_preds = model.predict(valid_generator)
val_preds_class_indices = np.argmax(val_preds, axis=1)

# True labels dari validation data
true_labels = valid_generator.classes

# Class labels
class_labels = list(valid_generator.class_indices.keys())

# Laporan klasifikasi
report = classification_report(true_labels, val_preds_class_indices, target_names=class_labels)
print(report)

# Plot training & validation loss and accuracy
plt.figure(figsize=(12, 4))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Loss')

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Accuracy')

plt.show()

# Save the model
model.save('resnet50_model_acc_95_99.h5')

from tensorflow.keras.preprocessing import image

def predict_image(uploaded_image_path):
    img = image.load_img(uploaded_image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict the class of the uploaded image
    predictions = model.predict(img_array / 255)
    predicted_class_index = np.argmax(predictions[0])
    confidence = predictions[0][predicted_class_index]
    
    # Get the class names
    class_names = train_generator.class_indices
    class_names = {v: k for k, v in class_names.items()}
    predicted_class_name = class_names[predicted_class_index]
    
    print(f"Prediction: {predicted_class_name} (Confidence: {confidence:.2f})")
    
    # Display the uploaded image
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class_name} \nConfidence: {confidence:.2f}")
    plt.axis('off')
    plt.show()

# Contoh penggunaan prediksi gambar
# predict_image('path_to_image.jpg')